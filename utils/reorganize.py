import SimpleITK as sitk
import os
from os.path import join
import shutil

def move_images(src_dir, dest_dir):
    
    for folder_name in os.listdir(src_dir):
        
        src_folder_name = join(src_dir,folder_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
        for img_name in os.listdir(join(src_dir,folder_name)):
            img_path = join(src_folder_name, img_name)
            sitk_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            # reoriented_img = sitk.ReorientImageFilter().Execute(sitk_img)
            output_path = join(dest_dir, f"{img_name.split('.')[0]}.nii.gz")
            sitk.WriteImage(sitk_img, output_path)
            print(output_path)
            print(1)
    return None

def move_files(src_dir, dest_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        # if file_name.startswith('CAD'):
        source = join(src_dir,file_name)
        destination = join(dest_dir,file_name)
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print(count,': copied', file_name)
            count+=1
    return None

def move_files_mca(src_dir, dest_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        source = join(src_dir,file_name)
        destination = join(dest_dir,f"MCA{file_name.split('.')[0][4:]}.nii.gz")
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print(count,': copied', destination)
            count+=1
    return None

def move_files_northwestern(src_dir, dest_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        source = join(src_dir,file_name)
        destination = join(dest_dir,f"NU_{file_name.split('.')[0]}.nii.gz")
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print(count,': copied', destination)
            count+=1
    return None

def move_files_nyu(src_dir, dest_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        if file_name.startswith('Patient'):
            source = join(src_dir,file_name)
            destination = join(dest_dir,f"nyu_{file_name.split('.')[0]}.nii.gz")
            # destination = join(dest_dir,f"nyu_{file_name.split('.')[0][:-5]}.nii.gz")
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print(count,': copied', destination)
                count+=1
    return None

def move_files_mcf(src_dir, dest_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        if file_name.startswith('MCF'):
            source = join(src_dir,file_name)
            destination = join(dest_dir,f"MCF_{file_name.split('.')[0][4:]}.nii.gz")
            # destination = join(dest_dir,f"nyu_{file_name.split('.')[0][:-5]}.nii.gz")
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print(count,': copied', destination)
                count+=1
    return None

def move_files_mayov1(src_dir, dest_dir):
    # fetch all files
    count=1
    for patient in os.listdir(src_dir):
        src_folder = join(src_dir,patient)
        for file_name in os.listdir(src_folder):
            if file_name.startswith('irene'):
                source = join(src_folder,file_name)
                destination = join(dest_dir,f"{patient}.nii.gz")
                # print('starts with irene,',file_name)
                if os.path.isfile(source):
                    shutil.copy(source, destination)
                    print(count,': copied', destination)
                    count+=1

    return None

def print_files(src_dir):
    # fetch all files
    count=1
    for file_name in os.listdir(src_dir):
        # if file_name.startswith('CAD'):
        source = join(src_dir,file_name)
        if os.path.isfile(source):
            print(count,file_name.split('.')[0])
            count+=1
    return None

# src_dir = "/data/Lanhong/Pancreas_radiomics/mayo_v1/mayo_v1_cropped_mask"
# dest_dir = "/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation"
# dest_dir = "/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation_t1"
# src_dir = "/data/Lanhong/Pancreas_radiomics/mayo_v1/mayo_v1_cropped_registeredT1"
# dest_dir = "/data/Lanhong/Pancreas_radiomics/MICCAI/t1/cropped"

src_dir = ['/data/datasets/backup_pancreas_ipmn_20230316/mayo_florida/gt_segmentation',
'/data/datasets/backup_pancreas_ipmn_20230316/mayo_florida/gt_segmentation_tmp']
for s in src_dir:
    print_files(s)




