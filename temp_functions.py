from PIL import Image
import numpy as np
import cv2

def read_no_whitespace(filePath, top_cuts = 10, side_cuts = 10, crop_thresh = 220,
                       isolatedPixelRemoval = 1, darken_thresh = 250, xy_size = None):
    """Read colour image into numpy array with fixed top- and side-cuts for the 
    original image. 'crop_thresh' indicates threshold pixel value to be cut out
    in cropping. 'xy_size' is the size of the image output defined as tuple"""
    
   
    # Load the image using PIL
    img = Image.open(filePath)
    
    # Convert the image to a numpy array
    img_array = np.array(img)
 
    ## Remove alpha-channel (if exists)
    if np.shape(img_array)[-1] == 4:
        # Split image into four channels
        b, g, r, a = cv2.split(img_array)
        # Build image excluding alpha
        img_array = cv2.merge((b,g,r))
    
    # Remove n rows from the top and bottom
    img_array = img_array[top_cuts:-top_cuts, :]
    
    # Remove m columns from the left and right
    img_array = img_array[:, side_cuts:-side_cuts]
    
    # Find the bounding box of the image without white space
    nonwhite_pixels = np.where(img_array < crop_thresh)
    bbox = (np.min(nonwhite_pixels[1]), np.min(nonwhite_pixels[0]), 
            np.max(nonwhite_pixels[1]), np.max(nonwhite_pixels[0]))
    
    # Use PIL to crop the image to the bounding box
    img = Image.fromarray(img_array).crop(bbox)
    
    # Convert the image back to a numpy array
    img = np.array(img)
    
    if isolatedPixelRemoval == 1:
        ## Remove remaining white rows (produced by isolated pixels)
        b, g, r = cv2.split(img)
        
        # Vertical
        v_mask = np.mean(b,1)<=crop_thresh
        b = b[v_mask,:]
        g = g[v_mask,:]
        r = r[v_mask,:]
        
        # Horizontal
        h_mask = np.mean(b,0)<=crop_thresh
        b = b[:, h_mask]
        g = g[:, h_mask]
        r = r[:, h_mask]
        
        # Put image back together
        img = cv2.merge((b,g,r))
        
    # Fill white areas with black if needed
    if darken_thresh is not None:
        black_pixels = np.where(
                (img[:, :, 0] >= darken_thresh) & 
                (img[:, :, 1] >= darken_thresh) & 
                (img[:, :, 2] >= darken_thresh)
            )
            # set those pixels to white
        img[black_pixels] = [0, 0, 0]

    # Resize image, if needed
    if xy_size is not None:
        img = cv2.resize(img, (xy_size[0], xy_size[1]))
    return img

def show_im(image, bw_inv=0):
        """Draw figure from array shaped image"""
        if bw_inv == 1:
            return Image.fromarray(cv2.bitwise_not(image))
        else:
            return Image.fromarray(image)
