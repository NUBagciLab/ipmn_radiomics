# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 07:57:52 2022

@author: kiliane
"""


import numpy as np
from collections import OrderedDict

import SimpleITK as sitk

import os
import shutil

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file) 


def convert_labels(in_file, out_file):
    #binary segmentation mask, convert label to 1
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    print('old labels are:',uniques)
    for u in uniques:
        if u not in [0, 1, 2, 3, 4]:
            print('label =',u)
            raise RuntimeError('unexpected label')
    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 1] = 1
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 3] = 1
    seg_new[img_npy == 4] = 1

    print('new labels are:', np.unique(seg_new))
    
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file) 
    
######################################################################

source_folder = "/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation_t1/cropped_variouslabel"
destination_folder = "/data/Lanhong/Pancreas_radiomics/MICCAI/gt_segmentation_t1/cropped"

count=1
# fetch all files
for file_name in os.listdir(source_folder):

    in_file = os.path.join(source_folder,file_name)
    out_file = os.path.join(destination_folder,file_name)
    convert_labels(in_file,out_file)
    print(count,':converted:', file_name)
    count+=1
        







