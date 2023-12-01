import SimpleITK as sitk
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) < 5:
    print("Usage: python bspline_registration.py <fixed_image_folder> <moving_image_folder> <number_of_iterations> <output_path>")
    sys.exit(1)

def command_iteration(method):
    print(f"Optimizer iteration: {method.GetOptimizerIteration()}, Metric value: {method.GetMetricValue()}")

fixed_path = sys.argv[1]
moving_path = sys.argv[2]
output_path = sys.argv[4]

# Convert number_of_iterations to integer
try:
    nb_iterations = int(sys.argv[3])
    if nb_iterations < 0:
        print("Error: number_of_iterations must be larger than 0.")
        sys.exit(1)
except ValueError:
    print("Error: number_of_iterations must be an integer.")
    sys.exit(1)

# Read the images
if fixed_path.endswith('.nii'):
    fixed_image = sitk.ReadImage(fixed_path)
else:
    fixed_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(fixed_path))

if moving_path.endswith('.nii'):
    moving_image = sitk.ReadImage(moving_path)
else:
    moving_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(moving_path))

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
transformDomainMeshSize = [16] * moving_resampled.GetDimension()
b_spline_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

# Set up the Registration Method
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsCorrelation()
registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=nb_iterations, maximumNumberOfCorrections=2, maximumNumberOfFunctionEvaluations=500, costFunctionConvergenceFactor=1e7)
registration_method.SetInitialTransform(b_spline_transform, True)
registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

try:
    # Execute Registration
    print("Starting registration...")
    final_transform = registration_method.Execute(fixed_image, moving_resampled)
    print("Registration completed.")

    # Resample the Moving Image
    print("Resampling moving image...")
    moving_resampled = sitk.Resample(moving_resampled, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_resampled.GetPixelID())

    # Write the images
    print("Writing output image...")
    sitk.WriteImage(moving_resampled, f"{output_path}.nii")
    print("Process completed successfully.")

except RuntimeError as e:
    print(f"Runtime error during registration: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")