"""
Sample code to render plant pointclouds.
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_plant(points, rgb, 
                image_size=256,
                background_color=(1, 1, 1),
                phi=0):
    device = get_device()
    points = points.to(device)
    rgb = rgb.to(device)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R ,T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=phi)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="plants_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--deg_res", type=int, default=5)   # Degree resolution for rotation steps

    args = parser.parse_args()

    data = load_rgbd_data()
    
    # Load Img1 Data
    img1_rgb = torch.tensor(data['rgb1'])
    img1_depth = torch.tensor(data['depth1'])
    img1_mask = torch.tensor(data['mask1'])
    img1_metadata = data['cameras1'] #Contains extrincs & intrinsics
    img1_pts, img1_rgba = unproject_depth_image(img1_rgb, img1_mask, img1_depth, img1_metadata)    
    img1_pts = img1_pts.unsqueeze(0)
    img1_rgb = (img1_rgba[:, :3]).unsqueeze(0)


    # Load Img2 Data
    img2_rgb = torch.tensor(data['rgb2'])
    img2_depth = torch.tensor(data['depth2'])
    img2_mask = torch.tensor(data['mask2'])
    img2_metadata = data['cameras2'] #Contains extrincs & intrinsics
    img2_pts, img2_rgba = unproject_depth_image(img2_rgb, img2_mask, img2_depth, img2_metadata)
    img2_pts = img2_pts.unsqueeze(0)
    img2_rgb = (img2_rgba[:, :3]).unsqueeze(0)
    

    # Combine Img1 & Img2 Points
    combined_pts = torch.cat((img1_pts, img2_pts), dim=1)
    combined_rgb = torch.cat((img1_rgb, img2_rgb), dim=1)

    # Render Plants
    image1 = render_plant(img1_pts, img1_rgb)
    image2 = render_plant(img2_pts, img2_rgb)
    image_combined = render_plant(combined_pts, combined_rgb)



    images1 = []
    images2 = []
    images_combined = []

    combined_frames = []


    import imageio

    for deg in range(0, 360 + args.deg_res, args.deg_res):
        image1 = render_plant(img1_pts, img1_rgb, phi=deg)
        image2 = render_plant(img2_pts, img2_rgb, phi=deg)
        image_combined = render_plant(combined_pts, combined_rgb, phi=deg)

        image1 = np.clip(image1, 0, 1)
        image2 = np.clip(image2, 0, 1)
        image_combined = np.clip(image_combined, 0, 1)


        image1_uint8 = (image1 * 255).astype(np.uint8)
        image2_uint8 = (image2 * 255).astype(np.uint8)
        image_combined_uint8 = (image_combined * 255).astype(np.uint8)

        stacked_image = np.hstack((image1_uint8, image2_uint8, image_combined_uint8))

        combined_frames.append(stacked_image)

    duration = int(1000 / len(combined_frames))

    imageio.mimsave(args.output_path, combined_frames, duration=duration / 1000)
    


