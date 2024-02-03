import os  # needed navigate the system to get the input data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import radiomics          # Radiomics package
from radiomics import featureextractor    
import SimpleITK as sitk
import nibabel as nib


def calculate_volume(imagePath,maskPath):
    img = nib.load(imagePath)
    seg = nib.load(maskPath)    

    header_img = img.header
    header_seg = seg.header
    zx, zy, zz = header_img["pixdim"][1:4]   #seg has the saome pixdim
    sx,sy,sz = header_seg["pixdim"][1:4] 
    if [zx,zy,zz]!=[sx,sy,sz]:
        print('Voxel size mismatch:', imagePath,'/n', maskPath)
        print(f'img voxel:{zx, zy, zz}, mask voxel:{sx,sy,sz}')
    voxel_vol_mm3 = zx * zy* zz

    ###image
    x, y, z = img.shape
    n_voxel = x*y*z
    body_vol_ml = (n_voxel * voxel_vol_mm3) / 1000

    ###segmentation
    seg_n_voxel = np.count_nonzero(seg.get_fdata())
    seg_vol_ml = (seg_n_voxel * voxel_vol_mm3) / 1000
    return body_vol_ml, seg_vol_ml


image_dir = "/data/Lanhong/Pancreas_radiomics/MICCAI/t2/reoriented"
# image_dir = "/data/datasets/pancreas_ipmn/nyu/t1"
mask_dir = "/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation/reoriented"

volume = np.array([['patient','volume']])
count = 1

### for nyu data structure


for patient in os.listdir(image_dir):
    patient = patient.split('.')[0]
    print('Extracting:',patient)
    # if patient.startswith('MCF'):
    imagePath = os.path.join(image_dir,patient+".nii.gz") # t2 & t1 are both compatible with this format
    maskPath = os.path.join(mask_dir,patient+".nii.gz") 

    if os.path.exists(imagePath) and os.path.exists(maskPath):
        _, seg_vol_ml = calculate_volume(imagePath, maskPath)
        volume = np.append(volume, [[patient,seg_vol_ml]],axis=0)
        print(count,': Calculated:',patient)
        count+=1

save_file_df = pd.DataFrame(volume[1:],columns=volume[0])
print(save_file_df)

save_file_df.to_csv('volume_complete.csv',index=False)
print('Done Bro')

 
# ### for mayo data structure
# for patient in os.listdir(image_dir):
#     print('Extracting:',patient)
#     patient_folder = os.path.join(image_dir,patient)
#     mask_folder= os.path.join(mask_dir,patient)

#     for filename in os.listdir(patient_folder):
#         if filename.startswith('greece'):
#             imagePath = os.path.join(patient_folder,filename) # .nii.gz
#     for filename_m in os.listdir(mask_folder):
#         # print('file',filename_m)
#         if filename_m.startswith('gr'):
#             maskPath = os.path.join(mask_folder,filename_m) 
#     for filename_m in os.listdir(mask_folder):
#         if filename_m.startswith('gr'):
#             print('?')
            
#     _, seg_vol_ml = calculate_volume(imagePath, maskPath)

#     volume = np.append(volume, [[patient,seg_vol_ml]],axis=0)
#     print(count,': Calculated:',patient)
#     count+=1