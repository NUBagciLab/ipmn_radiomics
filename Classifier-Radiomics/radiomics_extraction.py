import os
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import six

# Function to calculate pancreas volume for each patient

def calculate_volume(img_dir,mask_dir):
    volume = np.array([['patient','volume']])
    count = 1

    for patient in os.listdir(mask_dir):
        patient = patient.split('.')[0]
        print('Extracting:',patient)

        imagePath = os.path.join(img_dir,patient+".nii.gz") # t2 & t1 are both compatible with this format
        maskPath = os.path.join(mask_dir,patient+".nii.gz") 

        volume = np.array([['patient','volume']])
        count = 1
        if os.path.exists(imagePath) and os.path.exists(maskPath):
            img = nib.load(imagePath)
            seg = nib.load(maskPath)    

            header_img = img.header
            header_seg = seg.header
            zx, zy, zz = header_img["pixdim"][1:4]   #seg has the saome pixdim
            sx,sy,sz = header_seg["pixdim"][1:4] 
            if [zx,zy,zz]!=[sx,sy,sz]:
                raise Warning('Voxel size mismatch:', imagePath,'/n', maskPath)
            voxel_vol_mm3 = zx * zy* zz

            ###image
            x, y, z = img.shape
            n_voxel = x*y*z
            body_vol_ml = (n_voxel * voxel_vol_mm3) / 1000

            ###segmentation
            seg_n_voxel = np.count_nonzero(seg.get_fdata())
            seg_vol_ml = (seg_n_voxel * voxel_vol_mm3) / 1000

            volume = np.append(volume, [[patient,seg_vol_ml]],axis=0)
            print(count,': Calculated:',patient)
            count+=1
        df_volume = pd.DataFrame(volume[1:],columns=volume[0])
    return df_volume

# Function to extract radiomics features for each patient
def catch_features(imagePath,maskPath):
    if imagePath is None or maskPath is None:
        raise Exception('imagePath is None or maskPath is None!')
    image_1 = sitk.ReadImage(imagePath)
    label_1 = sitk.ReadImage(maskPath)
    # Display the images
    
    # Get the origins of the T2 and mask images
    t2_origin = image_1.GetOrigin()
    mask_origin = label_1.GetOrigin()
    t2_spacing = image_1.GetSpacing()
    mask_spacing = label_1.GetSpacing()

    # Check if the origins are different
    if not np.isclose(np.array(t2_origin), np.array(mask_origin),atol=1e-1).all():
        # Set the mask origin to be the same as the T2 origin
        label_1.SetOrigin(t2_origin)

    if not np.isclose(np.array(t2_spacing), np.array(mask_spacing),atol=1e-1).all():
        # Set the mask origin to be the same as the T2 origin
        label_1.SetSpacing(t2_spacing)
    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance=2e-1)
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()

    # label_1 = sitk.BinaryDilate(label_1, [5,5,5])  #Dilute as needed, no significant difference in this case.
    result_1 = extractor.execute(image_1, label_1)
    
    # Make an array of the values
    feature_val = np.array([])
    feature_name = np.array([])

    for key, value in six.iteritems(result_1):
        if key.startswith("original_"):
            feature_val = np.append(feature_val, value)
            feature_name = np.append(feature_name, key)
            
    return feature_val, feature_name

# Main function to concatenate volume and radiomics features and write to CSV
def main():

    #Extact radiomics t2
    image_dir = "/data/t2/preprocessed"
    mask_dir = "/data/gt_segmentation/cropped"

    n = 107
    save_file = np.empty([1,n])
    id_list = np.array([])
    patient_list = os.listdir(image_dir)
    count =1

    for patient in patient_list:
        print('Extracting:',patient)
        imagePath = os.path.join(image_dir,patient)
        maskPath= os.path.join(mask_dir,patient)
        if os.path.isfile(maskPath):
            try:
                features, name = catch_features(imagePath, maskPath)
            except:
                print(f'img{patient} cannot be extracted.')
                continue

            features = np.array(features).reshape([1,n])    
            id_list = np.append(id_list,patient.split('.')[0])
            np.concatenate(([[patient]],features),axis=1)
            save_file = np.append(save_file,features,axis=0)
            print(count,': Extracted:',patient)
            count+=1

    t2 = np.delete(save_file,0,0)
    df_t2 = pd.DataFrame(t2)
    df_t2.index = id_list
    df_t2.columns= name

    df_t2.to_csv('df_t2.csv')

    #Calculate volume
    image_dir = "/data/t2/raw"
    mask_dir = "/data/gt_segmentation/raw"
    df_volume = calculate_volume(image_dir,mask_dir)

    df_volume.to_csv('df_volume',index=False)
    
    #Read labels from CSV
    df_label = pd.read_csv('label.csv')

    #CONCAT
    df_t2.columns = ['t2_'+str(col) for col in df_t2.columns]  # add_prefix
    df_t1.columns = ['t1_'+str(col) for col in df_t1.columns]
    df_t2 = df_t2.rename(columns={'t2_patient': 'patient'})
    df_t1 = df_t1.rename(columns={'t1_patient': 'patient'})

    df_labeled = df_t1.merge(df_t2, on='patient')   #merge t2
    df_labeled = df_labeled.merge(df_volume, on ='patient')   #merge volume
    df_labeled = df_labeled.merge(df_label,on = 'patient')   

    # df_labeled = df_labeled[df_labeled.label<3]
    df_labeled.drop_duplicates(subset=['patient'],inplace=True)
    print(df_labeled)
    print(df_labeled['label'].value_counts())
    # Write final DataFrame to CSV
    df_labeled.to_csv('radiomics_NEW.csv', index=False)

if __name__ == '__main__':
    main()
