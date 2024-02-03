import SimpleITK as sitk
import numpy as np
import os
from os.path import join

def get_bounding_box(mask):
    # Find the bounding box of the mask
    binary_image = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=1)
    labeled_image = sitk.ConnectedComponent(binary_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(labeled_image, binary_image)
    x, y, z, w, h, d = stats.GetBoundingBox(1)
    
    # Perform a small dilation on the bounding box
    dilation = 0
    x, y, z, w, h, d = x-dilation, y-dilation, z-dilation, w+2*dilation, h+2*dilation, d+2*dilation
    
    return x, y, z, w, h, d

# Path to the folder containing the masks
folder_path = "/data/Lanhong/Pancreas_radiomics/mayo_v1_cropped_mask_radiomics"
dst_src = '/data/Lanhong/Pancreas_radiomics/bounding_box'
# Read all the masks in the folder
for file_name in os.listdir(folder_path):
    # Load the binary mask
    mask = sitk.ReadImage(os.path.join(folder_path, file_name))
    
    # Get the bounding box of the mask
    x, y, z, w, h, d = get_bounding_box(mask)
    
    # Draw the bounding box on the mask
    result = mask
    for i in range(x, x+w):
        for j in range(y, y+h):
            for k in range(z, z+d):
                result[i, j, k] = 1
    
    # Save the result
    sitk.WriteImage(result, join(dst_src,file_name))
    print(file_name)
