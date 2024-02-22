import numpy as np
import argparse
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
import torchvision.transforms
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader

import _init_paths

from configs import cfg, update_config
from dataset.build import build_dataset, get_dataloader
from utils.visualize import *
from utils.libmesh import check_mesh_contains
from utils.utils import load_camera_intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description='ArgumentParser for shapeExtractionNet')

    # general
    parser.add_argument('--cfg',
                        help='Experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--visualize',
                        help="Use this flag to turn on visualization",
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    # Don't use splits.csv to prepare for all models
    cfg.defrost()
    cfg.DATASET.SPLIT_CSV = "splits.csv"
    cfg.freeze()

    # -=-=-=-=-=- CREATE DATASET STRUCTURE -=-=-=-=-=- #
    dataset = build_dataset(cfg, 'all')
    all_loader = get_dataloader(cfg, split='all', output_mesh=True)
    device = "cuda"

    # Renderer
    sigma = 1e-6
    camera = load_camera_intrinsics(dataset.root_dir / cfg.DATASET.CAMERA)
    pcamera = FoVPerspectiveCameras(
        fov=camera['horizontalFOV'], degrees=False,
        device=device,
        zfar=1000,
    )
    raster_settings = RasterizationSettings(
        image_size=128, blur_radius=0.0,
    )

    blend_params = BlendParams(sigma=sigma)
    depth_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=pcamera, raster_settings=raster_settings),
        shader=HardDepthShader(
            device="cuda", cameras=pcamera, blend_params=blend_params
        ),
    )

    # additional rot matrix for aligning with camera
    Rz = torch.diag(
        torch.tensor([-1, -1, 1], dtype=torch.float32, device=device)
    ).unsqueeze(0)

    # iterate through all images
    for step, batch in tqdm(enumerate(all_loader), total=len(all_loader)):
        print(f"at step={step}")
        B = batch['image'].shape[0]

        gt_rot, gt_trans, gt_mesh, gt_masks, rgb = batch["rot"].to(device), batch["trans"].to(device), batch["mesh"].to(
            device), \
            batch['mask'].to(device), batch['image'].to(device)

        rot_render = torch.bmm(gt_rot.transpose(1, 2), Rz.repeat(gt_rot.shape[0], 1, 1)).detach()
        trans_render = gt_trans
        depths = depth_renderer(gt_mesh, R=rot_render, T=trans_render)
        depths[depths == 100] = 0
        masked_depths = depths * gt_masks.unsqueeze(-1)

        if args.visualize:
            for bid in range(B):
                plt.figure(figsize=(10, 10))
                plt.imshow((masked_depths[bid, ..., 0] / depths.max() * 255).cpu().numpy(), cmap=plt.cm.binary)
                plt.axis("off")
                plt.show()

                plt.figure(figsize=(10, 10))
                plt.imshow((torch.permute(rgb[bid, :3, ...], (1, 2, 0))).cpu().numpy())
                plt.axis("off")
                plt.show()
