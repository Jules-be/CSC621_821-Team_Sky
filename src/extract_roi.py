import SimpleITK as sitk
import numpy as np
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) < 4:
    print("Usage: python script_name.py <source_image_path> <segmented_image_path> <output_path>")
    sys.exit(1)

# Path to the original, segmented image and output files
source_image_path = sys.argv[1]
segmented_image_path = sys.argv[2]
output_path = sys.argv[3]

# Read the original and segmented images
source_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(source_image_path))
segmented_image = sitk.ReadImage(segmented_image_path)

source_np = sitk.GetArrayFromImage(source_image)
segmented_np = sitk.GetArrayFromImage(segmented_image)

# Apply the mask to retain the ROI and set the rest to the background value
roi_np = np.where(segmented_np == 255, source_np, 0)

# Convert the masked image to a SimpleITK image and save
roi_image = sitk.GetImageFromArray(roi_np)
output_path = f"{output_path}.nii"
sitk.WriteImage(roi_image, output_path)