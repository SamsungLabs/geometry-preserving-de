import os

MODEL_ZOO = ['mn_lrn4', 'lite0_lrn4', 'b0_lrn4', 'b1_lrn4', 'b2_lrn4', 'b3_lrn4', 'b4_lrn4', 'b5_lrn4']
DS_ZOO = {'nyu': 'd1', 'tum': 'd1', 'sintel': 'rel', 'eth3d': 'rel', 'diw': 'whdr'}

DATA_FOLDER = '/home/user01/Documents/work/our-repo/data-folder'

NYU_PATH = os.path.join(DATA_FOLDER, 'nyu_depth_v2_labeled.mat') # Path to .mat file
TUM_PATH = os.path.join(DATA_FOLDER, 'TUM')  # Folder with .h5 files
SINTEL_PATH = os.path.join(DATA_FOLDER, 'sintel')  # Should contain "final" and "depth" folders inside
DIW_PATH = os.path.join(DATA_FOLDER, 'DIW')  # Folder with "DIW_test.csv" and "DIW_test" subfolder
ETH3D_PATH = os.path.join(DATA_FOLDER, 'ETH3D')  # Folder with location subfolders
