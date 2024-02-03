
import os
import time
# file_path = '/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation_t1/reoriented/MCF_14.nii.gz'
# before = os.path.getmtime(file_path)


# # get the modified timestamp of the file after the overwrite
# after = os.path.getmtime(file_path)
# ref = os.path.getmtime('/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation_t1/reoriented/MCF_15.nii.gz')
# # compare the modified timestamps to see if the file was overwritten
# if before != after:
#     print("File was overwritten.")
#     print(before, after)
# else:
#     print("File was not overwritten.")
#     print('last modified:',time.ctime(after))

file_path = '/data/Lanhong/Pancreas_radiomics/MICCAI/t2/preprocessed/MCF_14.nii.gz'
after = os.path.getmtime(file_path)
print('last modified:',time.ctime(after))