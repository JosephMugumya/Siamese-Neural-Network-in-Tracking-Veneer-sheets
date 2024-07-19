
######rf#######
import sys
sys.path.append('E:Downloads\Thesis\Datasets\src')

#%% Import libraries
import os
import cv2
from temp_functions import read_no_whitespace

#%% Settings
# Image size used for gridding (x, y)
size = (924, 863)
# Square length and width in gridding (pixels)
gridPx = 300
# Threshold value for white colour to leave out
cropThr = 220
# Fixed cutting for top-below and right-left (in pixels)
topCut = 10
sideCut = 10

dataset_dir=r'E:\Downloads\Thesis\Datasets\src\dataset_original_2'
dest_dir='E:\Downloads\Thesis\Datasets\src\dataset_resized_2'

#%% Read pictures and save into destination folder
## 1) Removes whitespace around sheet image
## 2) Resizes before saving


wrong_pairs = [2321, 2278, 2486, 910, 2345,
                163, 1954, 2568, 2092, 2022,
                2131, 2276, 1889, 1890, 374]
defective_images = [2149, 1958, 2018, 712]
exclusions = set(wrong_pairs + defective_images)
exclusions_dry = ['dry_' + str(s) + '.png' for s in exclusions]
exclusions_wet = ['wet_' + str(s) + '.png' for s in exclusions]
exclusions = exclusions_dry + exclusions_wet

for condition in ['Dry', 'Wet']:
    print('Condition', condition, 'started')

    files = sorted(os.listdir(os.path.join(dataset_dir, condition)))

    for item in exclusions:
        if item in files:
            files.remove(item)

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for fname in files:
        img_path = os.path.join(dataset_dir, condition, fname)

        img = read_no_whitespace(img_path,
                                xy_size=size,
                                crop_thresh = cropThr,
                                top_cuts = topCut,
                                side_cuts = sideCut)
        # Change color format (otherwise will be shown blue)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ## Write image
        cv2.imwrite(os.path.join(dest_dir, condition, fname), img)
