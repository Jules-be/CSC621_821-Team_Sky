import SimpleITK as sitk

nii1 = sitk.ReadImage('data/results/moving_image_registered.nii')
nii2 = sitk.ReadImage('data/results/fixed_image.nii')

# Simple intensity-based blending
merged_image = (nii1 + nii2) / 2

# Save the merged image
sitk.WriteImage(merged_image, 'data/results/merged_output.nii')
