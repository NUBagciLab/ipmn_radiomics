import SimpleITK as sitk
import os


def reorient(stik_img, direction='RAS'):
    #print(name, stik_img.GetSize())
    #reoriented = sitk.DICOMOrient(stik_img, 'LPS')
    #reoriented = sitk.DICOMOrient(stik_img, 'RAS')
    #reoriented = sitk.DICOMOrient(stik_img, 'PLS') # default
    reoriented = sitk.DICOMOrient(stik_img, direction)

    return reoriented


input_folder = '/data/datasets/pancreas_ipmn/mayo_florida/t1/nii'
output_folder = '/data/datasets/pancreas_ipmn/mayo_florida/t1/reoriented'

count=1

for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder,filename)
    if os.path.isfile(filepath):
        img = sitk.ReadImage(filepath)
        img_reo = reorient(img)

        sitk.WriteImage(img_reo,os.path.join(output_folder,filename))
        print(f'{count}: reoriented: {filename}')
        count+=1
