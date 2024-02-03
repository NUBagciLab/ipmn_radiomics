import numpy as np
import nibabel as nib
import os
from os.path import join
import SimpleITK as sitk

def check_dim(dir1,dir2):
    # fetch all files
    print('These patient has a mismatch in the dimensions of mask and img:')
    count=1
    for file in os.listdir(dir2):
        img_path_raw = join(dir1,file)
        img_path_crop = join(dir2,file)

        if os.path.isfile(img_path_raw) & os.path.isfile(img_path_crop):
            img_r = nib.load(img_path_raw)
            img_c = nib.load(img_path_crop) 

            # get the direction cosine matrices of the two images
            dcm1 = np.array(img_r.shape)
            dcm2 = np.array(img_c.shape)

            # compare the direction cosine matrices using numpy's isclose function
            if not np.isclose(dcm1, dcm2,atol=1e-3).all():
                print(count,': ',file.split('.')[0])
                print(' orig dim=',img_r.shape, ', cropped dim=',img_c.shape )
                count+=1

        #     print(count,':',file, ' orig dim=',img_r.shape, ', cropped dim=',img_c.shape )
        #     count+=1
    return None

def check_orientation(dir1,dir2):
    # fetch all files
    print('These patient has a mismatch in the orientation of mask and img:')
    count=1
    for file in os.listdir(dir2):
        img_path_raw = join(dir1,file)
        img_path_crop = join(dir2,file)
        if os.path.isfile(img_path_raw) & os.path.isfile(img_path_crop):
            img1 = sitk.ReadImage(img_path_raw)
            img2 = sitk.ReadImage(img_path_crop) 
            
            # print(count,':',file, ' t2 orien=',img1.GetDirection(), ', mask orien=',img2.GetDirection() )
            
            # get the direction cosine matrices of the two images
            dcm1 = np.array(img1.GetDirection()).reshape(3, 3)
            dcm2 = np.array(img2.GetDirection()).reshape(3, 3)

            # compare the direction cosine matrices using numpy's isclose function
            if not np.isclose(dcm1, dcm2,atol=1e-1).all():
                print(count,': ',file.split('.')[0])
                print('orien1=',dcm1, '\norien2=',dcm2 )
                count+=1
            
    return None

def check_origin(dir1,dir2):
    # fetch all files
    print('These patient has a mismatch in the origin of mask and img:')
    count=1
    for file in os.listdir(dir2):
        img_path_raw = join(dir1,file)
        img_path_crop = join(dir2,file)
        if os.path.isfile(img_path_raw) & os.path.isfile(img_path_crop):
            img1 = sitk.ReadImage(img_path_raw)
            img2 = sitk.ReadImage(img_path_crop) 
            
            # print(count,':',file, ' t2 orien=',img1.GetDirection(), ', mask orien=',img2.GetDirection() )
            
            # get the direction cosine matrices of the two images
            dcm1 = np.array(img1.GetOrigin())
            dcm2 = np.array(img2.GetOrigin())

            # compare the direction cosine matrices using numpy's isclose function
            if not np.isclose(dcm1, dcm2,atol=1e-1).all():
                print(count,': ',file.split('.')[0])
                print('origin1=',dcm1, '\norigin2=',dcm2 )
                count+=1
            
    return None

def check_spacing(dir1,dir2):
    # fetch all files
    print('These patient has a mismatch in the spacing of mask and img:')
    count=1
    for file in os.listdir(dir2):
        img_path_raw = join(dir1,file)
        img_path_crop = join(dir2,file)
        if os.path.isfile(img_path_raw) & os.path.isfile(img_path_crop):
            img1 = sitk.ReadImage(img_path_raw)
            img2 = sitk.ReadImage(img_path_crop) 
            
            # print(count,':',file, ' t2 orien=',img1.GetDirection(), ', mask orien=',img2.GetDirection() )
            
            # get the direction cosine matrices of the two images
            dcm1 = np.array(img1.GetSpacing())
            dcm2 = np.array(img2.GetSpacing())

            # compare the direction cosine matrices using numpy's isclose function
            if not np.isclose(dcm1, dcm2,atol=1e-1).all():
                print(count,': ',file.split('.')[0])
                print('spacing1=',dcm1, '\nspacing2=',dcm2 )
                count+=1
            
    return None

def main():
    dir1 = '/data/Lanhong/Pancreas_radiomics/MICCAI/t1/DL/preprocessed'
    dir2 = '/data/Lanhong/Pancreas_radiomics/MICCAI/t2/preprocessed'
    check_dim(dir1,dir2)
    check_orientation(dir1,dir2)
    # check_origin(dir1,dir2)
    # check_spacing(dir1,dir2)

    # img_path_raw = '/data/Lanhong/Pancreas_radiomics/MICCAI/t2/reoriented/MCA40.nii.gz'
    # img_path_crop = '/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation/reoriented/MCA40.nii.gz'
    # img1 = sitk.ReadImage(img_path_raw)
    # img2 = sitk.ReadImage(img_path_crop) 
    # print('orien1=',img1.GetDirection(), '\norien2=',img2.GetDirection() )
            

if __name__ == '__main__':
    main()


