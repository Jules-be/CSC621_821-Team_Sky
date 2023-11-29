import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))

# Apply CurvatureFlowImageFilter for denoising
curvature_flow_filter = sitk.CurvatureFlowImageFilter()
curvature_flow_filter.SetNumberOfIterations(2)
curvature_flow_filter.SetTimeStep(0.05)
denoised_image = curvature_flow_filter.Execute(image)

# Rescale the intensity to 0-255
rescaled_image = sitk.RescaleIntensity(denoised_image, 0, 255)
rescaled_image = sitk.Cast(rescaled_image, sitk.sitkUInt8)

# Apply Canny Edge Detection
canny_edge = sitk.CannyEdgeDetection(sitk.Cast(rescaled_image, sitk.sitkFloat32), lowerThreshold=2, upperThreshold=10)
canny_edge = sitk.Cast(canny_edge, sitk.sitkUInt8)
canny_edge = canny_edge * 255

# Convert SimpleITK images to NumPy arrays
rescaled_array = sitk.GetArrayFromImage(rescaled_image)
canny_edge_array = sitk.GetArrayFromImage(canny_edge)

# Create a binary mask from the canny edge image (edges are 1, rest are 0)
edge_mask = canny_edge_array > 0

# Overlay the images
# Increase the intensity of edges for better visibility if needed
edge_intensity = 160
overlay_image = np.where(edge_mask, edge_intensity, rescaled_array)
overlay_image = sitk.GetImageFromArray(overlay_image)

sitk.WriteImage(overlay_image, 'data/results/canny_edges_patient_1.nii')

