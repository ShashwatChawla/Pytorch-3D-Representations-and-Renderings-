"""
Usage:
    python3 main.py --q {question_number}
    
"""

import argparse
import q_11
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        default='q_11',
        choices=["q_11", "q_12", "q_21", "q_22", "q_3", 
                 "q_4",  "q_51", "q_52", "q_53", "q_6", "q_7"],
    )
    args = parser.parse_args()

    if args.run == "q_11":
        print("#### 1.1. 360-degree Renders ####")
        subprocess.run(['python', 'q_11.py'])

    elif args.run == "q_12":
        print("####  1.2 Re-creating the Dolly Zoom ####")
        subprocess.run(['python', 'dolly_zoom.py'])

    elif args.run == "q_21":
        print("####  2.1 Constructing a Tetrahedron ####")
        subprocess.run(['python', 'render_tetrahedron.py'])

    elif args.run == "q_22":
        print("####  2.2 Constructing a Cube ####")
        subprocess.run(['python', 'render_cube.py'])

    elif args.run == "q_3":
        print("#### 3. Re-texturing a mesh ####")
        subprocess.run(['python', 'retexture_cow.py'])

    elif args.run == "q_4":
        print("#### 4. Camera Transformations####")
        # subprocess.run(['python', 'camera_transforms.py', '--transform', '1',])
        # subprocess.run(['python', 'camera_transforms.py', '--transform', '2',])
        # subprocess.run(['python', 'camera_transforms.py', '--transform', '3',])
        subprocess.run(['python', 'camera_transforms.py', '--transform', '4',])

    elif args.run == "q_51":
        print("####  5.1 Rendering Point Clouds from RGB-D Images ####")
        subprocess.run(['python', 'q_51.py'])
    
    elif args.run == "q_52":
        print("####  5.2 Parametric Functions ####")
        subprocess.run(['python', 'render_generic.py', "--render", "parametric", "--object", "torus"])
        # Chosen object is Cylinder
        subprocess.run(['python', 'render_generic.py', "--render", "parametric", "--object", "cylinder"])

    elif args.run == "q_53":
        print("####  5.3 Implicit Surfaces ####")
        subprocess.run(['python', 'render_generic.py', "--render", "implicit", "--object", "torus"])
        # Chosen object is Cylinder
        subprocess.run(['python', 'render_generic.py', "--render", "implicit", "--object", "cylinder"])
      
    elif args.run == "q_6":
        print("####  6. Do Something Fun ####")
        # Ensure this is run after downloading obj from: https://free3d.com/3d-model/realistic-tree-02-134612.html
        subprocess.run(['python', 'q_6.py'])
     
    elif args.run == "q_7":
        print("####  (Extra Credit) 7. Sampling Points on Meshes ####")
        subprocess.run(['python', 'sample_points_mesh.py'])

    
    
        
        







