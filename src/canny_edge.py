import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# Load the image
image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))

# Rescale the intensity to 0-255
rescaled_image = sitk.RescaleIntensity(image, 0, 255)

# Convert the image back to 16-bit
rescaled_image = sitk.Cast(rescaled_image, sitk.sitkInt16)

# # Apply Adaptive Histogram Equalization
# rescaled_image = sitk.AdaptiveHistogramEqualization(rescaled_image)

# Apply Canny Edge Detection
canny_edge = sitk.CannyEdgeDetection(sitk.Cast(rescaled_image, sitk.sitkFloat32), lowerThreshold=5, upperThreshold=30)
canny_array = sitk.GetArrayFromImage(canny_edge)
print("Data range:", canny_array.min(), canny_array.max())

# Save or display
sitk.WriteImage(canny_edge, 'data/results/canny_patient_1.nrrd')

# Select a slice to visualize, for example, the 5th slice
slice_number = 4  # Python uses 0-based indexing, so this is the 5th slice
selected_slice = canny_array[slice_number, :, :]

# Visualization
plt.figure()
plt.imshow(selected_slice, cmap='gray')
plt.colorbar()
plt.show()


