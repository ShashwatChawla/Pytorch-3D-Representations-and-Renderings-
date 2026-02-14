"""
Sample code to render cow for 360 degress and create gif
--> Adapted from starter/render_mesh.py

"""

import argparse

import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import torch
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

import os
import imageio 
import numpy as np


def get_cow_renderer(cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()
    

    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    return mesh, renderer

def render_cow(mesh, renderer, rotation, device=None,):

    if device is None:
        device = get_device()

    # Attempt New Cam rot/translation
    R_, t_ = pytorch3d.renderer.look_at_view_transform(dist=4, elev=0, azim=rotation)
    R_ = R_.to(device)
    t_ = t_.to(device)
    
    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_, T=t_, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="submission/q_11")
    parser.add_argument("--image_size", type=int, default=256)
    # Degree-resolution
    parser.add_argument("--deg_res", type=int, default=2)
    args = parser.parse_args()
    mesh, renderer = get_cow_renderer(cow_path=args.cow_path, image_size=args.image_size)

    my_images = []
    
    for deg in range(-360, 360+1, args.deg_res):
        
        image = render_cow(mesh, renderer, deg)
        my_images.append((image * 255).astype(np.uint8))
        
    duration = 1000//len(my_images)


    imageio.mimsave('cow_360.gif', my_images, duration=duration)
    print("#### Cow-360 Gif Created ###")
