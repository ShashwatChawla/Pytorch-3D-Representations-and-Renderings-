import argparse
import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import torch
import os
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


# Define Cube Vertices and Faces
def create_cube():
    '''
    Function to create cube verticex and faces
    '''
    vertices = torch.tensor([
        [-1.0, -1.0, -1.0],  # Vertex 0
        [1.0, -1.0, -1.0],   # Vertex 1
        [1.0, 1.0, -1.0],    # Vertex 2
        [-1.0, 1.0, -1.0],   # Vertex 3
        [-1.0, -1.0, 1.0],   # Vertex 4
        [1.0, -1.0, 1.0],    # Vertex 5
        [1.0, 1.0, 1.0],     # Vertex 6
        [-1.0, 1.0, 1.0],    # Vertex 7
    ], dtype=torch.float32)

    faces = torch.tensor([
        # Bottom face (z = -1)
        [0, 1, 2], [0, 2, 3],
        # Top face (z = +1)
        [4, 5, 6], [4, 6, 7],
        # Front face (y = +1)
        [3, 2, 6], [3, 6, 7],
        # Back face (y = -1)
        [0, 4, 5], [0, 5, 1],
        # Left face (x = -1)
        [0, 3, 7], [0, 7, 4],
        # Right face (x = +1)
        [1, 5, 6], [1, 6, 2],
    ], dtype=torch.int64)

    return vertices, faces

# Create Cube Mesh with Texture
def get_cube_renderer(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()

    # Get vertices and faces for the cube
    vertices, faces = create_cube()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)        # (N_f, 3) -> (1, N_f, 3)

    # Create a single-color texture for the mesh
    textures = torch.ones_like(vertices) * torch.tensor(color)  # (1, N_v, RGB)

    # Create a PyTorch3D Mesh object
    mesh = Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Create a renderer for rendering the mesh
    renderer = get_mesh_renderer(image_size=image_size)
    
    return mesh, renderer


# Render Cube from Different Angles
def render_cube(mesh, renderer, rotation_angle_deg=0, device=None):
    if device is None:
        device = get_device()

    R_, t_ = pytorch3d.renderer.look_at_view_transform(dist=4.0,
                                                       elev=30.0,
                                                       azim=rotation_angle_deg)
    R_ = R_.to(device)
    t_ = t_.to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R_, T=t_, fov=60.0,
                                                       device=device)

    lights = pytorch3d.renderer.PointLights(location=[[2.0, 2.0, -2.0]], device=device)

    # Render the mesh with the renderer and camera setup
    rend = renderer(meshes_world=mesh, cameras=cameras, lights=lights)
    
    rend = rend.cpu().numpy()[0, ..., :3]   # Convert tensor to numpy array (H x W x RGB)
    
    return rend

# Main Function to Render Tetrahedron from Multiple Angles and Save as GIF
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="cube_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--deg_res", type=int, default=5)   # Degree resolution for rotation steps
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get tetrahedron mesh and renderer
    mesh, renderer = get_cube_renderer(image_size=args.image_size, device=device)

    images = []
    
    import imageio

    for deg in range(0, 360 + args.deg_res, args.deg_res):
        image = render_cube(mesh=mesh,
                                   renderer=renderer,
                                   rotation_angle_deg=deg,
                                   device=device)
        images.append((image * 255).astype(np.uint8))

    duration = int(1000 / len(images))   # Duration per frame in milliseconds
    
    
    imageio.mimsave(args.output_path, images)

    print(f"GIF saved at {args.output_path}")

