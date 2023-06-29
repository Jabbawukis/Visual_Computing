import blenderproc as bproc
import numpy as np

bproc.init()

# Load the object
obj = bproc.loader.load_obj("/vol/fob-vol4/mi17/christod/Visual_Computing/exercises/exercise_05/Resources_DNN4VC_Synthetic/resources/lm/models/obj_000002.ply")

scale_factor = 100
light_position = np.array([2, -2, 0]) * scale_factor

light = bproc.types.Light()
light.set_location(light_position)
light.set_energy(300)

# Set the camera to be in front of the object
camera_matrix = np.array([[537.5, 0, 318.9], [0, 536.1, 238.4], [0, 0, 1]])
scaled_camera_matrix = camera_matrix / 2

cam_pose = bproc.math.build_transformation_mat([0, -5*scale_factor, 0], [np.pi / 2, 0, 0])
bproc.camera.add_camera_pose(cam_pose)
bproc.camera.set_intrinsics_from_K_matrix(scaled_camera_matrix, 320, 240)

# Render the scene
data = bproc.renderer.render()

# Write the rendering into an hdf5 file
bproc.writer.write_hdf5("output/", data)