import SimpleITK as sitk
import os
from os.path import join

def convert_images(src_dir, dest_dir):
    
    for folder_name in os.listdir(src_dir):
        
        dest_folder_name = join(dest_dir, folder_name)
        print(dest_folder_name)
        src_folder_name = join(src_dir,folder_name)
        if not os.path.exists(dest_folder_name):
            print(0.5)
            os.makedirs(dest_folder_name)
            
        for img_name in os.listdir(join(src_dir,folder_name)):
            img_path = join(src_folder_name, img_name)
            sitk_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            # reoriented_img = sitk.ReorientImageFilter().Execute(sitk_img)
            output_path = join(dest_folder_name, f"{img_name.split('.')[0]}.nii.gz")
            sitk.WriteImage(sitk_img, output_path)
            print(output_path)
            print(1)

src_dir = "/data/datasets/pancreas_old/orig/mayo_florida/v1/diagnosis_all/Diagnosis/Cropped_Scans/Cropped_Masks"
dest_dir = "/data/Lanhong/Pancreas_radiomics/mayo_v1_cropped_mask"

convert_images(src_dir, dest_dir)
