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
masked_slice = np.where(segmented_slice == 255, source_slice, 0)

# Assuming background pixels are 0
background_value = 0

# Identify non-background pixels
non_background_positions = np.where(masked_slice != background_value)

# Find the bounding box coordinates
min_y, max_y = np.min(non_background_positions[0]), np.max(non_background_positions[0])
min_x, max_x = np.min(non_background_positions[1]), np.max(non_background_positions[1])

# Crop the masked_slice
cropped_masked_slice = masked_slice[min_y:max_y+1, min_x:max_x+1]

# Plot the original and cropped side by side
plt.figure(figsize=(12, 6))

# Masked image slice
plt.subplot(1, 2, 1)
plt.imshow(masked_slice, cmap='gray', interpolation='none', aspect='equal')
plt.colorbar()
plt.title('Original Masked Slice')
plt.axis('off')

# Cropped image slice
plt.subplot(1, 2, 2)
plt.imshow(cropped_masked_slice, cmap='gray', interpolation='none', aspect='equal')
plt.colorbar()
plt.title("Cropped Masked Slice")
plt.axis('off')

plt.show()

glcm = graycomatrix(cropped_masked_slice.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Calculate texture properties
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
print(f"Contrast: {contrast:.2f}")
print(f"Dissimilarity: {dissimilarity:.2f}")
print(f"Homogeneity: {homogeneity:.2f}")

# Mean and Standard Deviation
mean = np.nanmean(masked_slice)
std_dev = np.nanstd(masked_slice)
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# Regularity (Smoothness)
variance = np.nanvar(masked_slice)
smoothness = 1 - (1 / (1 + variance))
print(f"Regularity (Smoothness): {smoothness:.2f}")

# Shannon Entropy
entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
print(f"Shannon Entropy: {entropy:.2f}")

# Uniformity (Energy)
uniformity = np.sum(glcm**2)
print(f"Uniformity (Energy): {uniformity:.2f}")