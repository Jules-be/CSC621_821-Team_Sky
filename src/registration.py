import SimpleITK as sitk

def command_iteration(method):
    print(f"Optimizer iteration: {method.GetOptimizerIteration()}, Metric value: {method.GetMetricValue()}")

# Read the images
fixed_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))
moving_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_2'))

# Convert the images to Float32
fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

# Adjusting the Image Spacing of the moving image to match with the fixed image
print("Fixed Image Spacing:", fixed_image.GetSpacing())
print("Moving Image Spacing:", moving_image.GetSpacing())

moving_image.SetSpacing(fixed_image.GetSpacing())
print("Adjusted Moving Image Spacing:", moving_image.GetSpacing())

# Then, use histogram matching to adjust the moving image to match the fixed image
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving_image = matcher.Execute(moving_image, fixed_image)

# Set up Euler Transform
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# Initialize B-Spline Transform
print("Setting up registration...")
transformDomainMeshSize = [16] * moving_image.GetDimension()
b_spline_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

# Set up the Registration Method
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsCorrelation()
registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=20, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000, costFunctionConvergenceFactor=1e7)
registration_method.SetInitialTransform(b_spline_transform, True)
registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

try:
    # Execute Registration
    print("Starting registration...")
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print("Registration completed.")

    # Resample the Moving Image
    print("Resampling moving image...")
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Write the images
    print("Writing output images...")
    sitk.WriteImage(fixed_image, 'data/results/fixed_image.nii')
    sitk.WriteImage(moving_resampled, 'data/results/moving_image_registered.nii')
    print("Process completed successfully.")

except RuntimeError as e:
    print(f"Runtime error during registration: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")