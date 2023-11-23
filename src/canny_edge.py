import SimpleITK as sitk
import matplotlib.pyplot as plt

# Load the image
image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))

# Denoise the image
denoised_image = sitk.GrayscaleDilate(image)

# Rescale the intensity to 0-255
rescaled_image = sitk.RescaleIntensity(denoised_image, 0, 255)

# Convert the image back to 16-bit
rescaled_image = sitk.Cast(rescaled_image, sitk.sitkInt16)

# Apply Canny Edge Detection
canny_edge = sitk.CannyEdgeDetection(sitk.Cast(rescaled_image, sitk.sitkFloat32), lowerThreshold=5, upperThreshold=20)
canny_array = sitk.GetArrayFromImage(canny_edge)
print("Data range:", canny_array.min(), canny_array.max())

# Save or display
sitk.WriteImage(canny_edge, 'data/results/canny_patient_1.nrrd')

# Select a slice to visualize
slice_number = 4
selected_slice = canny_array[slice_number, :, :]

# Visualization
plt.figure()
plt.imshow(selected_slice, cmap='gray')
plt.colorbar()
plt.show()


