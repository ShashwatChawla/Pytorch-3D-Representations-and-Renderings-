import torch
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform, 
    FoVPerspectiveCameras, 
    PointsRasterizationSettings, 
    PointsRenderer, 
    PointsRasterizer, 
    AlphaCompositor
)
import imageio
import numpy as np
import argparse
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def sample_points_on_mesh(mesh, num_samples):
    device = mesh.device
    vertices = mesh.verts_packed()  
    faces = mesh.faces_packed()    

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    
    face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)  
    face_probs = face_areas / face_areas.sum()
    sampled_face_idxs = torch.multinomial(face_probs, num_samples, replacement=True)
    sampled_faces = faces[sampled_face_idxs]

    u, v = torch.rand(num_samples, 1, device=device), torch.rand(num_samples, 1, device=device)
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    w = 1 - (u + v)

    sampled_points = (
        vertices[sampled_faces[:, 0]] * u +
        vertices[sampled_faces[:, 1]] * v +
        vertices[sampled_faces[:, 2]] * w
    )
    
    return sampled_points


def get_points_renderer(image_size=256, background_color=(1, 1, 1), device=None):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color)
    )
    return renderer.to(device)


def render_point_cloud(points, rgb, renderer, azim, device):
    point_cloud = Pointclouds(points=[points], features=[rgb])
    
    R, T = look_at_view_transform(dist=4, elev=0, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    
    images = renderer(point_cloud, cameras=cameras)
    return images[0, ..., :3].cpu().numpy()

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
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_gif", type=str, default="cow_mesh_points.gif")
    parser.add_argument("--deg_res", type=int, default=5)
    args = parser.parse_args()

    # Load cow mesh
    cow_mesh = load_objs_as_meshes([args.cow_path], device=device)
   
    # Renderer
    renderer = get_points_renderer(image_size=args.image_size, device=device)

    sample_sizes = [10, 100, 1000, 10000]
    sampled_points_list = [sample_points_on_mesh(cow_mesh, size) for size in sample_sizes]
    rgb_color = torch.tensor([[0.7, 0.7, 1]], device=device)
    rgb_list = [torch.ones_like(points) * rgb_color for points in sampled_points_list]


    mesh_, renderer_ = get_cow_renderer(cow_path=args.cow_path, image_size=args.image_size)


    images = []
    for azim in range(0, 360, args.deg_res):
        rendered_images = []
        
        cow_image = render_cow(mesh_, renderer_, azim)
        cow_image_uint8 = (cow_image * 255).astype(np.uint8)

        rendered_images.append(cow_image_uint8)

        for points, rgb in zip(sampled_points_list, rgb_list):
            image = render_point_cloud(points, rgb, renderer, azim, device)
            image_uint8 = (image * 255).astype(np.uint8)
            rendered_images.append(image_uint8)
        
        # Stack all images horizontally
        stacked_image = np.hstack(rendered_images)
        images.append(stacked_image)

    # Save GIF
    imageio.mimsave(args.output_gif, images, duration=0.1)
    print(f"GIF saved at {args.output_gif}")