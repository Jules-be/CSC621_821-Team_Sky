import SimpleITK as sitk
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) < 4:
    print("Usage: python dilation.py <folder_path> <radius> <output_path>")
    sys.exit(1)

image_path = sys.argv[1]
radius = int(sys.argv[2])
output_path = sys.argv[3]

# Read image
image = sitk.ReadImage(image_path)

# Convert image to binary format if necessary
image = sitk.Cast(image, sitk.sitkFloat32) / 255.0
binary_image = sitk.BinaryThreshold(image, lowerThreshold=0.5, upperThreshold=1.5, insideValue=1, outsideValue=0)

# Create a binary dilation filter
dilate_filter = sitk.BinaryDilateImageFilter()
dilate_filter.SetKernelType(sitk.sitkBall)
dilate_filter.SetKernelRadius(radius)

# Apply dilation
dilated_image = dilate_filter.Execute(binary_image)
dilated_image = dilated_image * 255

# Save the result
sitk.WriteImage(dilated_image, f'{output_path}.nii')
