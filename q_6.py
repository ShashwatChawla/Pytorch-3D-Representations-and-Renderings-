import argparse
import os
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

# Function to load seasonal textures
def get_seasonal_texture(vertices, rotation, device):
    """
    Function to change the tree's texture based on the rotation.
    Simulates changing seasons from Spring/Summer to Autumn to Winter.
    """
    green = torch.tensor([0.0, 0.5, 0.0])  # Green color for spring/summer
    yellow_orange = torch.tensor([1.0, 0.7, 0.0])  # Yellow-orange for autumn
    bare = torch.tensor([0.5, 0.3, 0.0])  # Bare branches for winter (brownish)

    if 0 <= rotation < 90:  # Spring/Summer
        texture_color = green
    elif 90 <= rotation < 180:  # Summer
        texture_color = green 
    elif 180 <= rotation < 270:  # Autumn
        texture_color = yellow_orange
    else:  # Winter
        texture_color = bare
    
    textures = torch.ones_like(vertices) * texture_color.to(device)  # (1, N_v, 3)
    return textures

def get_tree_rendered(tree_path="data/cow_on_plane.obj", image_size=256, device=None):
    if device is None:
        device = get_device()
    
    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = load_cow_mesh(tree_path)
    vertices = vertices.unsqueeze(0).to(device)  # (N_v, 3) -> (1, N_v, 3), move to correct device
    faces = faces.unsqueeze(0).to(device)  # (N_f, 3) -> (1, N_f, 3), move to correct device

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(torch.ones_like(vertices).to(device) * torch.tensor([0.0, 1.0, 0.0]).to(device)),
    )

    mesh = mesh.to(device)

    return mesh, renderer, vertices  

def render_tree(mesh, renderer, rotation, vertices, device=None):
    if device is None:
        device = get_device()

    # Set camera view based on rotation (azimuth)
    R_, t_ = pytorch3d.renderer.look_at_view_transform(dist=15, elev=25, azim=rotation)
    R_ = R_.to(device)
    t_ = t_.to(device)

    # Prepare the camera
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_, T=t_, fov=60, device=device
    )

    # Update the tree's texture depending on the rotation (season)
    seasonal_texture = get_seasonal_texture(vertices, rotation, device)

    mesh.textures = pytorch3d.renderer.TexturesVertex(seasonal_texture.to(device))

    light_color = torch.tensor([[1.0, 1.0, 1.0]], device=device)  

    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -3]], device=device, ambient_color=light_color, diffuse_color=light_color
    )

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, default="data/59-tree01/Tree01/3dmodel/tree01.obj")
    parser.add_argument("--image_size", type=int, default=256)
    # Degree-resolution
    parser.add_argument("--deg_res", type=int, default=2)
    args = parser.parse_args()

    # Load the tree mesh and renderer
    mesh, renderer, vertices = get_tree_rendered(tree_path=args.tree_path, image_size=args.image_size)

    # List to hold the rendered images for the GIF
    my_images = []

    # Generate images by rotating the tree and varying the light (seasonal effect)
    for deg in range(0, 360 + args.deg_res, args.deg_res):
        image = render_tree(mesh, renderer, deg, vertices)
        my_images.append((image * 255).astype(np.uint8))

    # Save the images as a GIF
    duration = 1000 // len(my_images)
    imageio.mimsave('tree_360.gif', my_images, duration=duration)

    print("#### Gif Created ###")
