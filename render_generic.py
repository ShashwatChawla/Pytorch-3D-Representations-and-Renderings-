"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
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

def render_cylinder(image_size=256, num_samples=200, radius=1.0, height=2.0, device=None, deg=0):
    """
    Renders a cylinder using parametric sampling.
    radius: Radius of the cylinder
    height: Height of the cylinder
    """
    if device is None:
        device = get_device()

    # Parameter ranges
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    z = torch.linspace(-height / 2, height / 2, num_samples)

    # Create a grid of theta and z
    Theta, Z = torch.meshgrid(theta, z, indexing='ij')

    # Parametric equations for cylinder
    x = radius * torch.cos(Theta)
    y = radius * torch.sin(Theta)

    # Stack into (N, 3) point cloud
    points = torch.stack((x.flatten(), y.flatten(), Z.flatten()), dim=1)

    # Normalize for color mapping
    color = (points - points.min()) / (points.max() - points.min())

    # Create the point cloud structure
    cylinder_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

     # Set up camera
    R ,T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=145, azim=deg)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    # Set up camera
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 5]], device=device)

    # Get renderer (ensure you have a `get_points_renderer` function)
    renderer = get_points_renderer(image_size=image_size, device=device)

    # Render the cylinder
    rend = renderer(cylinder_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus(image_size=256, num_samples=200, R=1.0, r=0.4, device=None, deg=0):
    """
    Renders a torus using parametric sampling.
    R: Major radius (distance from the center of the tube to the center of the torus)
    r: Minor radius (radius of the tube)
    """
    if device is None:
        device = get_device()

    # Parameter ranges
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)

    # Create a grid of phi and theta
    Phi, Theta = torch.meshgrid(phi, theta, indexing='ij')

    # Parametric equations for torus
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    # Stack into (N, 3) point cloud
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Normalize for color mapping
    color = (points - points.min()) / (points.max() - points.min())

    # Create the point cloud structure
    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # Set up camera
    R ,T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=deg)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)

    # Get renderer (ensure you have a `get_points_renderer` function)
    renderer = get_points_renderer(image_size=image_size, device=device)

    # Render the torus
    rend = renderer(torus_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus_mesh(image_size=256, voxel_size=64, R=1.0, r=0.4, device=None, deg=0):
    import mcubes
    
    if device is None:
        device = get_device()
    
    min_value = -(R + r + 0.1)
    max_value = R + r + 0.1
    
    X, Y, Z = torch.meshgrid([
        torch.linspace(min_value, max_value, voxel_size),
        torch.linspace(min_value, max_value, voxel_size),
        torch.linspace(min_value, max_value, voxel_size)
    ])

    # Implicit function for a torus
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2

    # Apply Marching Cubes
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels.numpy()), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Normalize vertex coordinates
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    # Apply color textures based on vertex positions
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    # Create Mesh object
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    
    # Lighting and Camera setup
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R_cam, T_cam = pytorch3d.renderer.look_at_view_transform(dist=3, elev=30, azim=45)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R_cam, T=T_cam, device=device)

    # Set up camera
    R ,T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=30, azim=deg)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)



    # Render the mesh
    rend = renderer(mesh, cameras=cameras, lights=lights)
    
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_cylinder_mesh(image_size=256, voxel_size=64, r=0.5, device=None, deg=0):
    if device is None:
        device = get_device()

    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    
    # Implicit equation for the cylinder
    voxels = X**2 + Y**2 - r**2  # Cylinder along z-axis, radius r

    # Apply marching cubes to extract the mesh surface
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Renormalize the vertices for visualization
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    # Lighting and camera setup
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=145, azim=0)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    R ,T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=145, azim=deg)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the mesh
    rend = renderer(mesh, cameras=cameras, lights=lights)

    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="parametric",
        choices=["parametric", "implicit"],
    )
    # parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--deg_res", type=int, default=5)   # Degree resolution for rotation steps

    parser.add_argument("--object", type=str, default="torus")
    args = parser.parse_args()


    import imageio
    import os
    image = None
    images = []

    for deg in range(0, 360 + args.deg_res, args.deg_res):
        if args.render == "parametric":
            if args.object == "torus":
                image = render_torus(image_size=args.image_size, num_samples=args.num_samples, deg=deg)
            elif args.object == "cylinder":
                image = render_cylinder(image_size=args.image_size, num_samples=args.num_samples, deg=deg)
            
        elif args.render == "implicit":
            if args.object == "torus":
                image = render_torus_mesh(image_size=args.image_size, deg=deg)
            elif args.object == "cylinder":
                image = render_cylinder_mesh(image_size=args.image_size, deg=deg)
    
        else:
            raise Exception("Did not understand {}".format(args.render))
        

        image = (image * 255).astype(np.uint8)
        images.append(image)

        duration = int(1000 / len(images))

    path_ = os.path.join(f"{args.render}_{args.object}_360.gif")
    imageio.mimsave(path_, images, duration=duration / 1000)

    print(f"Gif saved at {path_}")

