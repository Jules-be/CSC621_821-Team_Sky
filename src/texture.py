import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import graycomatrix, graycoprops

# Check if the script received the image path as an argument
if len(sys.argv) < 3:
    print("Usage: python script_name.py <roi_image_path> <slice_index>")
    sys.exit(1)

# Path to the ROI image file
roi_image_path = sys.argv[1]

# Convert slice_index to integer
try:
    slice_idx = int(sys.argv[2])
    if slice_idx < 0 or slice_idx > 9:
        print("Error: slice_index must be between 0 and 9.")
        sys.exit(1)
except ValueError:
    print("Error: slice_index must be an integer.")
    sys.exit(1)

# Load the ROI image
if roi_image_path.endswith('.nii'):
    roi_image = sitk.ReadImage(roi_image_path)
else:
    roi_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(roi_image_path))

# Getting the numpy array from the image
roi_np = sitk.GetArrayFromImage(roi_image)

# Extract the specific slice
roi_slice = roi_np[slice_idx, :, :]

# Assuming background pixels are 0
background_value = 0

# Identify non-background pixels
non_background_positions = np.where(roi_slice != background_value)

# Find the bounding box coordinates
min_y, max_y = np.min(non_background_positions[0]), np.max(non_background_positions[0])
min_x, max_x = np.min(non_background_positions[1]), np.max(non_background_positions[1])

# Crop the roi_slice
cropped_roi_slice = roi_slice[min_y:max_y+1, min_x:max_x+1]

# Plot the original and cropped side by side
plt.figure(figsize=(12, 6))

# Original image slice
plt.subplot(1, 2, 1)
plt.imshow(roi_slice, cmap='gray', interpolation='none', aspect='equal')
plt.colorbar()
plt.title('Original ROI Slice')
plt.axis('off')

# Cropped image slice
plt.subplot(1, 2, 2)
plt.imshow(cropped_roi_slice, cmap='gray', interpolation='none', aspect='equal')
plt.colorbar()
plt.title("Cropped ROI Slice")
plt.axis('off')

plt.show()

glcm = graycomatrix(cropped_roi_slice.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Calculate texture properties and store in a list
texture_properties = [
    ('Contrast', f"{graycoprops(glcm, 'contrast')[0, 0]:.3f}"),
    ('Dissimilarity', f"{graycoprops(glcm, 'dissimilarity')[0, 0]:.3f}"),
    ('Homogeneity', f"{graycoprops(glcm, 'homogeneity')[0, 0]:.3f}"),
    ('Mean', f"{np.nanmean(cropped_roi_slice):.3f}"),
    ('Standard Deviation', f"{np.nanstd(cropped_roi_slice):.3f}"),
    ('Regularity (Smoothness)', f"{1 - (1 / (1 + np.nanvar(cropped_roi_slice))):.3f}"),
    ('Shannon Entropy', f"{-np.sum(glcm * np.log2(glcm + (glcm == 0))):.3f}"),
    ('Uniformity (Energy)', f"{np.sum(glcm**2):.3f}")
]

# Create a figure for the table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

# Create and display the table
table = ax.table(cellText=texture_properties, colLabels=['Metric', 'Value'], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.show()
