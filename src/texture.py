import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import graycomatrix, graycoprops

# Check if the script received the folder path as an argument
if len(sys.argv) < 4:
    print("Usage: python script_name.py <source_image_path> <segmented_image_path> <slice_index>")
    sys.exit(1)

# Path to the original and segmented image files
source_image_path = sys.argv[1]
segmented_image_path = sys.argv[2]

# Convert slice_index to integer
try:
    slice_idx = int(sys.argv[3])
    if slice_idx < 0 or slice_idx > 9:
        print("Error: slice_index must be between 0 and 9.")
        sys.exit(1)
except ValueError:
    print("Error: slice_index must be an integer.")
    sys.exit(1)

# Read the original and segmented images
source_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(source_image_path))
segmented_image = sitk.ReadImage(segmented_image_path)

source_np = sitk.GetArrayFromImage(source_image)
segmented_np = sitk.GetArrayFromImage(segmented_image)

# Extract the specific slice from both images
source_slice = source_np[slice_idx, :, :]
segmented_slice = segmented_np[slice_idx, :, :]

# Apply the mask
masked_slice = np.where(segmented_slice == 1, source_slice, 0)

glcm = graycomatrix(masked_slice.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Calculate texture properties
print("Contrast:", graycoprops(glcm, 'contrast'))
print("Dissimilarity:", graycoprops(glcm, 'dissimilarity'))
print("Homogeneity:", graycoprops(glcm, 'homogeneity'))