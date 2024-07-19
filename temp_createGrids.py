#%% Import libraries
import os
import cv2

#%% Function definitions
def show_im(image, bw_inv=0):
        """Draw figure from array shaped image"""
        if bw_inv == 1:
            return Image.fromarray(cv2.bitwise_not(image))
        else:
            return Image.fromarray(image)

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


# Folders
pfolder = '...'   #Source wet
dfolder = '...'   # - " - dry
pPicDest = '...'  #Destination wet
dPicDest = '...'  # - " - dry

# List files in the source folders
pfiles = os.listdir(pfolder)
dfiles = os.listdir(dfolder)

#%% TEST reading a sample picture
img = read_no_whitespace(dfolder+dfiles[16], 
                         xy_size=(924, 863), 
                         darken_thresh = None)
show_im(img)

#%% Read pictures and save into destination folder
## 1) Removes whitespace around sheet image
## 2) Resizes before saving

## Peeling sheets
for n in range(0,len(pfiles)):
    # Show progress
    print(str(n)+'/'+str(len(pfiles)))
    # Read and resize
    im = read_no_whitespace(pfolder+pfiles[n], 
                            xy_size=size,
                            crop_thresh = cropThr,
                            top_cuts = topCut, 
                            side_cuts = sideCut)
    # Change color format (otherwise will be shown blue)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ## Write image
    cv2.imwrite(
        pPicDest+pfiles[n].split('.')[0]+'.png', 
        im
        )

## Drying sheets
for n in range(0,len(dfiles)):
    # Show progress
    print(str(n)+'/'+str(len(dfiles)))
    # Read and resize
    im = read_no_whitespace(dfolder+dfiles[n], 
                            xy_size=size,
                            crop_thresh = cropThr,
                            top_cuts = topCut, 
                            side_cuts = sideCut)
    # Change color format (otherwise will be shown blue)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ## Write image
    cv2.imwrite(
        dPicDest+dfiles[n].split('.')[0]+'.png', 
        im
        )


#%% Use equally-sized images to create gridded images into memory
# List images (written in the previous section)
pFiles = os.listdir(pPicDest)
dFiles = os.listdir(dPicDest)

# Create lists for storing image grids
p_testing = []
d_testing = []

# Create datasets (equal number of p- and d-sheets is assumed)
for x in range(0, len(pfiles):
    print(x)
    # Append gridded peeling images into list
    p_testing.append(resize_and_crop(pPicDest+pFiles[x], grid_size = gridPx, numpyOutput = 1))
    # Append gridded drying images into list
    d_testing.append(resize_and_crop(dPicDest+dFiles[x], grid_size = gridPx, numpyOutput = 1))

#%% TEST that grids are stored ok
# Show first grid image of the first peeling sheet
show_im(p_testing[0][0])

# -"- drying sheet
show_im(d_testing[0][0])
