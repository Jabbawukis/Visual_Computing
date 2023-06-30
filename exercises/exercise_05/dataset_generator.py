import blenderproc as bproc
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', nargs='?', help="Path to the 3d objects")
parser.add_argument('--cc_textures_path', nargs='?', default="resources/cctextures",
                    help="Path to downloaded cc textures")
parser.add_argument('--output_dir', nargs='?', help="Path to where the final files will be saved ")
parser.add_argument('--num_samples', nargs='?', type=int, default=20, help="Number of samples")
parser.add_argument('--start_index', nargs='?', type=int, default=0, help="First image")

args = parser.parse_args()

bproc.init()

"""
TASK 1: Add scaled camera matrix here and set intrinsic matrix and resolution (320x240)
"""
camera_matrix = np.array([[537.5, 0, 318.9], [0, 536.1, 238.4], [0, 0, 1]])
scaled_camera_matrix = camera_matrix / 2  # Scale camera matrix for 320x240 resolution
bproc.camera.set_intrinsics_from_K_matrix(scaled_camera_matrix, 320, 240)

# load a random sample of objects into the scene
target_objs = bproc.loader.load_bop_objs(bop_dataset_path=args.mesh_path,
                                         mm2m=True,
                                         sample_objects=True,
                                         num_of_objs_to_sample=1,
                                         obj_ids=[2])

"""
TASK 2: Load distractor objects 
- Use all obj_ids (1-15) besides the one of the target object.
- Sample 10 objects
- Place the objects on the surface (later in code) and make sure that the camera puts the target_obj in the image center.
- Make sure that the new objects are shaded similar to the original objects

# distractor objs = 
"""

distractor_objs = bproc.loader.load_bop_objs(bop_dataset_path=args.mesh_path,
                                             mm2m=True,
                                             sample_objects=True,
                                             num_of_objs_to_sample=10,
                                             obj_ids=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
for obj in distractor_objs:
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(target_objs):
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

"""
TASK 3: Complete the room by adding the missing room planes.

The room should consits of 4 walls and a floor (no ceiling needed), add the three missing walls with correct location and rotation.

"""
# create room (currently it is no room yet)
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 0], rotation=[0, 1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 0], rotation=[0, -1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                               elevation_min=5, elevation_max=89, uniform_volume=False)
light_point.set_location(location)

# sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)


# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))


# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=target_objs + distractor_objs,
                                                      surface=room_planes[0],
                                                      sample_pose_func=sample_initial_pose,
                                                      min_distance=0.2,
                                                      max_distance=0.5)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

poses = 0
while poses < args.num_samples:
    # Sample location

    """
    TASK 4: Sample the location unsing blenderprocs shell sampler (used above already). 

    - The distance of the camera and the origin (0,0,0) should be between 0.6 and 1.24m.
    - The elevation angle should not be flatter than 30 degree but also not steeper than 80 degree.
    - The sampler should sample the angle and radius uniformly.
    """
    # See task 4. We need a sampler here instead of a fixed location.
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.6, radius_max=1.24, elevation_min=30,
                                   elevation_max=80)
    # location = [0, -1.0, 0]
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi([placed_objects[0]])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1

"""
TASK 5: Activate semantic segmentation "enable_segmentation_output". 
Add a custom key to all objects (also planes) with a bool value specifying if they get a label. 
Only the target should get a labeled. 
Use the function 'set_cp' and map by the key during data generation. 
"""
bproc.renderer.enable_segmentation_output(map_by=["target"], default_values={'category_id': 0, 'target': False})
# Add custom property to objects
for obj in placed_objects:
    if obj not in target_objs:
        obj.set_cp("target", False)
    else:
        obj.set_cp("target", True)
for obj in room_planes:
    obj.set_cp("target", False)

# sets the maximum number of samples to render for each pixel
bproc.renderer.set_max_amount_of_samples(1)

# render the whole pipeline
data = bproc.renderer.render()

"""
TASK 6: Save the dataset

- Create folder "masks" and "img" in args.output_dir.
- Iterate over the keys 'target_segmaps' and 'colors' in data and save the outputs as png images.
- Images should be named as the test data (four digit numbers)
- Save images with PIL and use mode "L" for masks to make sure that the output has only a single channel.
- Save color images as RGB.
- Save masks with values 0 (background) and 255 (object).
- Use args.start_index to specify the first filename. We have to call this script multiple times and don't want that existing images are overwritten.
"""

os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "img"), exist_ok=True)

# Save the dataset
for i, (target_segmap, color) in enumerate(zip(data['target_segmaps'], data['colors'])):
    # Generate file names with leading zeros
    mask_file_name = f"{args.start_index + i:04d}.png"
    color_file_name = f"{args.start_index + i:04d}.png"

    target_segmap[target_segmap == 1] = 255

    unique_numbers, counts = np.unique(target_segmap, return_counts=True)

    print(unique_numbers)
    print(counts)

    # Save the mask image
    mask_image = Image.fromarray(target_segmap.astype(np.uint8), mode='L')
    mask_image.save(os.path.join(args.output_dir, "masks", mask_file_name))

    # Save the color image
    color_image = Image.fromarray(color, mode='RGB')
    color_image.save(os.path.join(args.output_dir, "img", color_file_name))
