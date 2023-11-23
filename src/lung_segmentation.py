import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) < 3:
    print("Usage: python script_name.py <folder_path> <slice_index>")
    sys.exit(1)

# Retrieve the folder path from the command line argument
folder_path = sys.argv[1]

# Convert slice_index to integer
try:
    slice_idx = int(sys.argv[2])
    if slice_idx < 0 or slice_idx > 9:
        print("Error: slice_index must be between 0 and 9.")
        sys.exit(1)
except ValueError:
    print("Error: slice_index must be an integer.")
    sys.exit(1)

# Load the image
image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(folder_path))

# Denoise the image
denoised_image = sitk.GrayscaleDilate(image)

# Rescale the intensity to 0-255
rescaled_image = sitk.RescaleIntensity(denoised_image, 0, 255)

# Convert the SimpleITK image to a NumPy array
np_image = sitk.GetArrayFromImage(rescaled_image)

# Function to handle mouse click event
def onclick(event):
    global seed_points, count
    ix, iy = event.xdata, event.ydata
    seed_points.append((int(ix), int(iy), slice_idx))
    count += 1
    print(f"Point {count}: ({ix}, {iy})")

    # Disconnect after 4 points are selected
    if count == 4:
        fig.canvas.mpl_disconnect(cid)

# Display an image slice
plt.imshow(np_image[slice_idx], cmap='gray')
plt.title('Click to select 4 points')
plt.axis('off')

seed_points = []
count = 0
fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Flatten the array to 1D for histogram plotting
flattened_image_array = np_image.flatten()

confidence_connected = sitk.ConfidenceConnected(rescaled_image, seedList=seed_points, numberOfIterations=10, multiplier=2, initialNeighborhoodRadius=1, replaceValue=1)

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
plt.imshow(sitk.GetArrayFromImage(segmented_slice), cmap='Blues')
plt.title('Segmented Image Slice')
plt.axis('off')

plt.show()

# After your segmentation process
segmented_image = confidence_connected
print("Segmentation pixel values:", np.unique(sitk.GetArrayFromImage(segmented_image)))
segmented_np = sitk.GetArrayFromImage(segmented_image)
segmented_image = sitk.GetImageFromArray(segmented_np)
output_filename = 'data/results/segmented_image.nii'
sitk.WriteImage(segmented_image, output_filename)