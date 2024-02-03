import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def plot_bin_center(data_path,output_path,center):
    # Get a list of all the image files in the folder
    # image_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')]
    image_files = [f for f in os.listdir(data_path) if f.startswith(f'{center}')]

    # Loop through the images and plot their histograms
    fig, ax = plt.subplots()
    plt.ylim([0, 150000])

    count=1
    for img_file in image_files:
        # Load the image
        img = sitk.ReadImage(os.path.join(data_path, img_file))
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.flatten()

        # Plot the histogram
        ax.hist(img_array, bins=50, alpha=0.5, label=img_file)
        print(f'{count}: plotted: {img_file}')
        count+=1

    # Add a legend to the plot
    ax.legend()

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Pixel Values')
    ax.set_ylabel('Frequency')

    # Show the plot
    # plt.show()
    plt.savefig(output_path)
    return None

def plot_bin_all(data_path,output_path):
    # Get a list of all the image files in the folder
    # image_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')]
    image_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')]

    # Loop through the images and plot their histograms
    fig, ax = plt.subplots()
    plt.ylim([0, 150000])

    count=1
    for img_file in image_files:
        # Load the image
        img = sitk.ReadImage(os.path.join(data_path, img_file))
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.flatten()

        # Plot the histogram
        ax.hist(img_array, bins=50, alpha=0.5, label=img_file)
        print(f'{count}: plotted: {img_file}')
        count+=1

    # Add a legend to the plot
    # ax.legend()

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Pixel Values')
    ax.set_ylabel('Frequency')

    # Show the plot
    # plt.show()
    plt.savefig(output_path)
    return None

def plot_line(img_dir,output_path):
    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.ylim([0, 150000])
    # Iterate over all image files in the directory
    count=1
    for img_file in os.listdir(img_dir):
        if img_file.endswith('.nii.gz'):
            # Load the image using SimpleITK
            img = sitk.ReadImage(os.path.join(img_dir, img_file))
            
            # Get the pixel values as a numpy array
            arr = sitk.GetArrayFromImage(img)
            
            # Compute the histogram
            hist, bin_edges = np.histogram(arr.flatten(), bins=100)
            
            # Compute the bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot the histogram as a line
            ax.plot(bin_centers, hist, label=img_file)
            print(f'{count}: plotted: {img_file}')
            count+=1
            
    # Add labels and legend
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    # ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(output_path)

    return None

def main():

    # Set the path to the folder containing the images
    # for center in ['nyu','NU','AHN','MCF','MCA']:
    #     data_path = '/data/Lanhong/Pancreas_radiomics/MICCAI/t1/cropped'
    #     output_path = f'histograms_cropped_{center}.png'
    #     output_path_all = 'histograms_cropped.png'
    #     plot_bin_center(data_path,output_path,center)

    data_path = '/data/Lanhong/Pancreas_radiomics/MICCAI/t1/DL/cropped'
    output_path_all = 'line_cropped_t1_new.png'
    print(f'There are {len(os.listdir(data_path))} within the input folder.') 
    plot_line(data_path,output_path_all)
    return None


if __name__ == '__main__':
    main()