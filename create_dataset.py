"""
This module is used for creating a cross-validated dataset.
"""
########rf###############
import sys
sys.path.append('E:\Downloads\Thesis\Datasets\src')

import os
import random
import numpy as np
import cv2
from create_image_grids import resize_and_crop

def create_dataset(dataset_dir=r'E:\Downloads\Thesis\Datasets\src\dataset_resized_test',
                   dest_dir=r'E:\Downloads\Thesis\Datasets\src\data_cross_new')-> None:
    """
    This function is used for creating a cross-validated dataset.
    """


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

        files_shuf = []
        if condition == 'Dry':
            index_shuf = list(range(len(files)))
            random.shuffle(index_shuf)
        for i in index_shuf:
            files_shuf.append(files[i])

        folds = np.array_split(files_shuf, 5)

        for i in range(5):
            print('Split', i, 'started')
            test_data = folds[i]
            #train_data = [item for sublist in np.delete(folds, i, 0) for item in sublist]
            train_data = np.delete(folds, i, 0).flatten()

            # Square length and width in gridding (pixels)
            gridPx = 224

            for fname in test_data:
                img_path = os.path.join(dataset_dir, condition, fname)

                img_grids = resize_and_crop(img_path, grid_size = gridPx, numpyOutput = 1)

                for j in range(len(img_grids)): # pylint: disable=consider-using-enumerate

                    dest_path = os.path.join(dest_dir, 'Split{}', 'Test', condition).format(i)
                    if not os.path.exists(dest_dir):
                        os.mkdir(dest_path)
                    ## Write image
                    img_grid = cv2.cvtColor(img_grids[j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(dest_path,
                                             fname.split('.')[0] + '_' + str(j) + '.png'),
                                img_grid)

            for fname in train_data:
                img_path = os.path.join(dataset_dir, condition, fname)

                img_grids = resize_and_crop(img_path, grid_size = gridPx, numpyOutput = 1)

                for j in range(len(img_grids)): # pylint: disable=consider-using-enumerate

                    dest_path = os.path.join(dest_dir, 'Split{}', 'Train', condition).format(i)
                    if not os.path.exists(dest_dir):
                        os.mkdir(dest_path)
                    ## Write image
                    img_grid = cv2.cvtColor(img_grids[j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(dest_path,
                                             fname.split('.')[0] + '_' + str(j) + '.png'),
                                img_grid)

if __name__ == "__main__":
    create_dataset()
