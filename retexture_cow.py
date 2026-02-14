"""
Sample code to re-texture a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
    color1 = torch.tensor([0, 0, 1]), color2 = torch.tensor([1, 0, 0]), phi=0
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    
    # Get nearest and farthest points
    z_values = vertices[:, 2]
    z_min = torch.min(z_values)
    z_max = torch.max(z_values)
    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    # textures = torch.ones_like(vertices)  # (1, N_v, 3)
    
    # Texture a mesh


    alpha = (z_values - z_min)/ (z_max - z_min)
    alpha = (alpha.unsqueeze(1).expand(-1, 3)).unsqueeze(0)

    textures = alpha*color2 + (1 - alpha)*color1
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    R_, t_ = pytorch3d.renderer.look_at_view_transform(dist=4.0,
                                                       elev=30.0,
                                                       azim=phi)
    R_ = R_.to(device)
    t_ = t_.to(device)



    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R_, T=t_, fov=60.0,
                                                       device=device)


    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="cow_retexture_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--deg_res", type=int, default=5)   # Degree resolution for rotation steps

    args = parser.parse_args()

    images = []
    import numpy as np
    import imageio

    color1 = torch.tensor([1, 1, 0]) 
    color2 = torch.tensor([0, 1, 1])
    for deg in range(0, 360 + args.deg_res, args.deg_res):
        image = render_cow(cow_path=args.cow_path, image_size=args.image_size, 
                           color1=color1, color2=color2, phi=deg,
)
        images.append((image * 255).astype(np.uint8))

    duration = int(1000 / len(images))   # Duration per frame in milliseconds

    imageio.mimsave(args.output_path, images)

    print(f"GIF saved at {args.output_path}")
