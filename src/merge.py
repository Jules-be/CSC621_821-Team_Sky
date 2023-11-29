import SimpleITK as sitk
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) != 4:
    print("Usage: python merge.py <fixed_image_path> <co_registered_moving_image_path> <output_path>")
    sys.exit(1)

fixed_path = sys.argv[1]
moving_path = sys.argv[2]
output_path = sys.argv[3]

print(moving_path)
# Read the images
fixed_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames('data/covid_negative_data/patient_1'))
moving_image = sitk.ReadImage('data/results/moving_image_registered_10_iter.nii')

# Read the images
if fixed_path.endswith('.nii'):
    fixed_image = sitk.ReadImage(fixed_path)
else:
    fixed_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(fixed_path))

if moving_path.endswith('.nii'):
    moving_image = sitk.ReadImage(moving_path)
else:
    moving_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(moving_path))


# Cast both images to the same pixel type (e.g., 32-bit float)
fixed_image_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
moving_image_float = sitk.Cast(moving_image, sitk.sitkFloat32)

# Resample the moving image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image_float)
resampler.SetInterpolator(sitk.sitkLinear)
resampled_moving_image = resampler.Execute(moving_image_float)

# Blend the images
blended_image = (fixed_image_float + resampled_moving_image) / 2

# Write the blended image to a file
sitk.WriteImage(blended_image, f'{output_path}.nii')
