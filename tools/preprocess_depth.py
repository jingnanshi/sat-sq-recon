import numpy as np
import argparse

import torch
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
    TexturesVertex,
)
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader

import _init_paths

from configs import cfg, update_config
from dataset.build import build_dataset, get_dataloader
from utils.visualize import *
from utils.libmesh import check_mesh_contains
from utils.utils import load_camera_intrinsics


class HardNOCSShader(torch.nn.Module):
    """ Shader that ignores lighting to render NOCS values """

    def __init__(
            self, device="cpu", cameras=None, blend_params=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardNOCSShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


def make_NOCS_vertex_textures(meshes):
    """ Make NOCS texture for rendering NOCS maps """
    mesh_verts = meshes.verts_list()
    with torch.no_grad():
        nocs_tex_tensor = torch.stack(mesh_verts).contiguous().clone()
        # shift by 0.5 b/c model is normalized within [-0.5, 0.5]
        # we need the nocs map be within [0, 1]
        nocs_tex_tensor = nocs_tex_tensor + 0.5
    nocs_textures = TexturesVertex(verts_features=nocs_tex_tensor)
    return nocs_textures


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
    if path.name.split(".")[-1].lower() != "png":
        raise ValueError("Only PNG format is currently supported.")

    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def save_nocs(path, im):
    """Saves a NOCS image (16-bit) to a PNG file.

    :param path: Path to the output NOCS image file.
    :param im: ndarray with the NOCS image to save. Dimension: (H, W, C).
    """
    if path.name.split(".")[-1].lower() != "png":
        raise ValueError("Only PNG format is currently supported.")

    im_uint16 = np.round(im).astype(np.uint16)
    im_list = im_uint16.reshape(-1, im_uint16.shape[1] * im_uint16.shape[2]).tolist()

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_nocs = png.Writer(width=im.shape[1], height=im.shape[0], greyscale=False, bitdepth=16)

    with open(path, "wb") as f:
        w_nocs.write(f, im_list)


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
    # do not scale the image down
    cfg.DATASET.IMAGE_SIZE = (256, 256)
    cfg.TRAIN.WORKERS = 5
    cfg.freeze()

    # -=-=-=-=-=- CREATE DATASET STRUCTURE -=-=-=-=-=- #
    dataset = build_dataset(cfg, 'all')
    all_loader = get_dataloader(cfg, split='all', output_mesh=True)
    zfar = 1000
    image_size = 256
    depth_scale = 65536 / zfar
    nocs_scale = 65536
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
        image_size=image_size, blur_radius=0.0,
    )

    blend_params = BlendParams(sigma=sigma)
    depth_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=pcamera, raster_settings=raster_settings),
        shader=HardDepthShader(
            device="cuda", cameras=pcamera, blend_params=blend_params
        ),
    )

    nocs_blend_params = BlendParams(sigma=sigma, background_color=(0, 0, 0))
    nocs_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=pcamera, raster_settings=raster_settings),
        shader=HardNOCSShader(
            device="cuda", cameras=pcamera, blend_params=nocs_blend_params,
        ),
    )

    # additional rot matrix for aligning with camera
    Rz = torch.diag(
        torch.tensor([-1, -1, 1], dtype=torch.float32, device=device)
    ).unsqueeze(0)

    # iterate through all images
    for step, batch in tqdm(enumerate(all_loader), total=len(all_loader)):
        B = batch['image'].shape[0]

        gt_rot, gt_trans, gt_mesh, gt_masks, rgb, model_idxs, base_filenames = batch["rot"].to(device), batch[
            "trans"].to(device), batch["mesh"].to(device), batch['mask'].to(device), \
            batch['image'].to(device), batch['model_idx'], batch['base_filename']

        # nocs rendering
        with torch.no_grad():
            # rotation and translation for rendering
            rot_render = torch.bmm(gt_rot.transpose(1, 2), Rz.repeat(gt_rot.shape[0], 1, 1)).detach()
            trans_render = gt_trans

            nocs_textures = make_NOCS_vertex_textures(gt_mesh)
            nocs_mesh = gt_mesh.clone()
            nocs_mesh.textures = nocs_textures
            nocs = nocs_renderer(nocs_mesh, R=rot_render, T=trans_render)
            nocs[:, :, :, -1] = gt_masks

            # depth rendering
            # (N, H, W, K)
            depths = depth_renderer(gt_mesh, R=rot_render, T=trans_render)

            # reset unknown depths to zero instead of zfar
            depths[depths == zfar] = 0

        if args.visualize:
            masked_depths = depths * gt_masks.unsqueeze(-1)
            masked_nocs = nocs[:, :, :, :3] * gt_masks.unsqueeze(-1)
            denorm_rgb = rgb * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
            for bid in range(B):
                plt.figure(figsize=(10, 10))
                plt.imshow((masked_depths[bid, ..., 0] / depths.max() * 255).cpu().numpy(), cmap=plt.cm.binary)
                plt.axis("off")
                plt.show()

                plt.figure(figsize=(10, 10))
                plt.imshow((torch.permute(denorm_rgb[bid, :3, ...], (1, 2, 0))).cpu().numpy())
                plt.axis("off")
                plt.show()

                plt.figure(figsize=(10, 10))
                plt.imshow((masked_nocs[bid, ..., :3]).cpu().numpy())
                plt.axis("off")
                plt.show()

        # save nocs and depths
        processed_depths = depths * depth_scale
        processed_nocs = nocs[:, :, :, :3] * nocs_scale
        processed_nocs[processed_nocs < 0] = 0
        processed_nocs[processed_nocs > 65536] = 65536

        for bid in range(B):
            bname = base_filenames[bid]

            # save depth
            path_to_depth = all_loader.dataset.datasets[model_idxs[bid]].path_to_depth_dir
            depth_path = path_to_depth / f"{base_filenames[bid]}.png"
            save_depth(depth_path, processed_depths[bid, ..., 0].cpu().numpy())

            # save nocs
            path_to_nocs = all_loader.dataset.datasets[model_idxs[bid]].path_to_nocs_dir
            nocs_path = path_to_nocs / f"{base_filenames[bid]}.png"
            save_nocs(nocs_path, processed_nocs[bid, ..., :3].cpu().numpy())