"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    # parser.add_argument("--output_path", type=str, default="images/")
    parser.add_argument("--transform", type=int, default=1) # To enable multiple transformations
    args = parser.parse_args()
    
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    import os
    
    R_rel = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
    T_rel = [0, 0, 0]
    print(f"### Transforming Action {args.transform}")
    if args.transform == 1:    
        rotation = R.from_euler('z', -90, degrees = True)
        R_rel = rotation.as_matrix().tolist()
        
    elif args.transform == 2:    
        T_rel = [0, 0, 1]

    elif args.transform == 3:
        rotation = R.from_euler('y', 5, degrees = True)    
        R_rel = rotation.as_matrix().tolist()
        T_rel = [0.2, 0, 0]


    elif args.transform == 4:
        rotation = R.from_euler('Y', 90, degrees = True)    
        R_rel = rotation.as_matrix().tolist()
        # According to the rotated frame
        T_rel = [-3, 0, 3]
        


    transformed_img = render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, 
                        R_relative=R_rel, T_relative=T_rel, device=get_device())
    

    path_ = os.path.join(f"cow_transformed_{args.transform}.jpg")
    plt.imsave(path_, transformed_img)
    print(f"Image saved at {path_}")
