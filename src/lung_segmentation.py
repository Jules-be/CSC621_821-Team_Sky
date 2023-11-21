import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["SITK_SHOW_COMMAND"] = "/Users/jules/Desktop/Fiji.app/Contents/MacOS/ImageJ-macosx"

T1_WINDOW_LEVEL = (512,5)

# Load the image
image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))

# Rescale the intensity to 0-255
rescaled_image = sitk.RescaleIntensity(image, 0, 255)

# Select the seed point
seed_points = [(146, 156, 0), (140, 340, 0), (411, 336, 0), (437, 281, 0)]

# Convert the SimpleITK image to a NumPy array
np_image = sitk.GetArrayFromImage(rescaled_image)

# Flatten the array to 1D for histogram plotting
flattened_image_array = np_image.flatten()

confidence_connected = sitk.ConfidenceConnected(rescaled_image, seedList=seed_points, numberOfIterations=10, multiplier=2, initialNeighborhoodRadius=1, replaceValue=1)

slice_idx = 0

# Extract the 2D slice from the 3D rescaled image
rescaled_slice = rescaled_image[:, :, slice_idx]

# Extract the 2D slice from the 3D segmented image
segmented_slice = confidence_connected[:, :, slice_idx]

# Plot the original and segmented images side by side
plt.figure(figsize=(10, 5))

# Original image slice
plt.subplot(1, 2, 1)
plt.imshow(sitk.GetArrayFromImage(rescaled_slice), cmap='gray')
plt.title('Original Image Slice')
plt.axis('off')

# Segmented image slice
plt.subplot(1, 2, 2)
plt.imshow(sitk.GetArrayFromImage(segmented_slice), cmap='magma')
plt.title('Segmented Image Slice')
plt.axis('off')

plt.show()