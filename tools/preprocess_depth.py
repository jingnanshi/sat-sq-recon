import numpy as np
import argparse
import trimesh
from tqdm import tqdm
import png
import matplotlib.pyplot as plt

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
import torchvision.transforms
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRendererWithFragments,
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


def depth_to_point_cloud_map_batched(depth, intrinsics, grid_x, grid_y):
    """Return point cloud in (B, 3, H, W)"""
    B = intrinsics.shape[0]
    # calcualte the x, y (lots of broadcasting)
    x = (
            (grid_x[None, None, ...] - intrinsics[:, 0, 0, 2].reshape((B, 1, 1, 1)))
            * depth
            / intrinsics[:, 0, 0, 0].reshape(B, 1, 1, 1)
    )
    y = (
            (grid_y[None, None, ...] - intrinsics[:, 0, 1, 2].reshape((B, 1, 1, 1)))
            * depth
            / intrinsics[:, 0, 1, 1].reshape(B, 1, 1, 1)
    )
    pts = torch.cat((x, y, depth), dim=1)
    return pts


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if path.split(".")[-1].lower() != "png":
        raise ValueError("Only PNG format is currently supported.")

    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


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
    zfar = 1000
    device = "cuda"

    # Renderer
    sigma = 1e-6
    camera = load_camera_intrinsics(dataset.root_dir / cfg.DATASET.CAMERA)
    pcamera = FoVPerspectiveCameras(
        fov=camera['horizontalFOV'], degrees=False,
        device=device,
        zfar=zfar,
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

        # depth rendering
        rot_render = torch.bmm(gt_rot.transpose(1, 2), Rz.repeat(gt_rot.shape[0], 1, 1)).detach()
        trans_render = gt_trans
        depths = depth_renderer(gt_mesh, R=rot_render, T=trans_render)

        # reset unknown depths to zero instead of zfar
        depths[depths == zfar] = 0

        if args.visualize:
            masked_depths = depths * gt_masks.unsqueeze(-1)
            for bid in range(B):
                plt.figure(figsize=(10, 10))
                plt.imshow((masked_depths[bid, ..., 0] / depths.max() * 255).cpu().numpy(), cmap=plt.cm.binary)
                plt.axis("off")
                plt.show()

                plt.figure(figsize=(10, 10))
                plt.imshow((torch.permute(rgb[bid, :3, ...], (1, 2, 0))).cpu().numpy())
                plt.axis("off")
                plt.show()

        # TODO: get NOCS
        pc = depth_to_point_cloud_map_batched(
            torch.permute(depths, (2, 0, 1)),
            torch.as_tensor(state["camera"]["K"]).unsqueeze(0).unsqueeze(0),
            grid_x=depth_map_grid_x,
            grid_y=depth_map_grid_y,
        )

        # TODO: save depth images
        depth_scale = 65536 / zfar
        processed_depths = depths * depth_scale

        # TODO: save NOCS images

        # for bid in range(B):
        #    save_depth(None, depths[bid, ..., 0].cpu().numpy())
