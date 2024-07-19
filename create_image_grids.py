"""TODO: add docstring"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image

def rotate_image(image, angle):
    """Rotate image using given angle"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, 
                            borderValue=(255,255,255))
    return result

def resize_img(image, px_size = 400):
    """Resize image stored as numpy array. From\
    https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv"""
    if px_size is None:
        return image
    r = px_size / image.shape[1]
    dim = (px_size, int(image.shape[0] * r))
    # perform the actual resizing of the image and show it
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def crop_image(image, pixel_value=255):
    """Crop image (whitespace excluded)"""
    crop_rows = image[~np.all(image == pixel_value, axis=1), :]
    cropped_image = crop_rows[:, ~np.all(crop_rows >= pixel_value, axis=0)]
    return cropped_image

def crop_colour_image(image, pixel_value=255, blurring = 5):
    """Crop image (whitespace excluded)"""
    # Blur image
    img = cv2.medianBlur(image,blurring)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find the contours of the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (the object) in the image
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask from the largest contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop the image to remove the white space
    x, y, w, h = cv2.boundingRect(largest_contour)
    result = result[y:y+h, x:x+w]
    return result

def resize_and_crop(image_path, grid_size, numpyOutput = 1):
    """Distributes given image into a grid of [grid_size] * [grid_size] * 3"""
    # Create a lookup list for image partitioning into desired grid size
    lookup_list = [grid_size*x for x in range(1, 5000)]
    with Image.open(image_path) as image:
        # Calculate image sizing
        xsize = min(lookup_list, key=lambda x:abs(x-image.size[0]))
        ysize = min(lookup_list, key=lambda x:abs(x-image.size[1]))
        size = (xsize, ysize)
        # Resize the image to concur with the given grid size
        image = image.resize(size)
        
        # Grid size (square)
        part_size = (grid_size, grid_size)

        # Crop the image into equally sized parts
        parts = []
        for y in range(0, size[1], part_size[1]):
            for x in range(0, size[0], part_size[0]):
                part = image.crop((x, y, x + part_size[0], y + part_size[1]))
                parts.append(part)
        if numpyOutput == 1:
            # Convert PIL-images into numpy image arrays inside the list
            parts = [np.array(parts[x]) for x in range(len(parts))]
            # Convert list of numpy image arrays into single numpy array
            parts = np.array(parts)
        return parts

def read_rotate_resize(filePath, size=800, cropping = 1, top_cuts = 0, 
                       side_cuts = 0, rot_rng = [-10, 10], rot_decim = 4, bw=1,
                       crop_thresh = 255, ret_full_img = 0):
    """Read raw image file to numpy array and resize into desired size"""
    
    #Read image in black-white, leave out whitespace and return resized
    img = cv2.imread(filePath,0)
    #Crop out whitespaces from the original image
    if cropping == 1:
        img = crop_image(img[top_cuts:-1-top_cuts,side_cuts:-1-side_cuts], pixel_value=crop_thresh)
    
    #Rotate cropped image
    angles = []
    angles.append(list(np.arange(rot_rng[0], rot_rng[1]+1, 1)))
    if rot_decim > 0:
        angles.append(list(np.arange(-9,10, 1)/10))
    if rot_decim > 1:
        angles.append(list(np.arange(-9,10, 1)/100))
    if rot_decim > 2:
        angles.append(list(np.arange(-9,10, 1)/1000))
    if rot_decim > 3:
        angles.append(list(np.arange(-9,10, 1)/10000))

    #Calculate mean whiteness for rotation using small image
    img_small = resize_img(img, px_size=size)
    
    #Rotate to desired decimals
    for x, _ in enumerate(angles):
        if x == 0:
            rot0 = 0
        else:
            rot0 = rot_opt
        rots = [np.mean(crop_image(rotate_image(img_small, rot0+angles[x][n]))) 
                for n in range(len(angles[x]))]
        rots = pd.DataFrame([angles[x], rots], index=['angle', 'bright']).transpose()
        rots['angle'] = rots['angle']+rot0
        # Get rotation angle
        try:
            rot_opt = round(rots.loc[rots['bright'] == min(rots['bright'])]['angle'],5).item()
        except:
            pass
    # Set final rotation angle
    rot_1 = rot_opt
    print('Rotated '+str(round(rot_1, rot_decim)))
    # 1 Case for small image output
    if ret_full_img == 0:
        if rot_1!=0:
            #Rotate full sized image
            img = crop_image(rotate_image(img, rot_1))
        if bw == 0:
            return resize_img(cv2.imread(filePath), px_size = size)
        else:
            return crop_image(resize_img(img, px_size = size))
    # 2 Case for full image output 
    if ret_full_img == 1:
        img = cv2.imread(filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rot_1!=0:
            #Rotate full sized image
            img = crop_colour_image(rotate_image(img, rot_1))
        if bw == 0:
            return img
        else:
            return crop_colour_image(img)



#%% Example use
#im = read_rotate_resize(
#                        'MySourceFile', 
#                        side_cuts = 5, 
#                        top_cuts = 5, 
#                        crop_thresh = 200,
#                        ret_full_img = 1)
## Write image
# im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
# cv2.imwrite(
#      'my_resultFileDestination'+'.png', 
#      im
#      )
## Read (again and make image grids with example sizes 32x32 and 64x64)
# test = resize_and_crop('my_resultFileDestination', 32)
# test2 = resize_and_crop('my_resultFileDestination', 64)
