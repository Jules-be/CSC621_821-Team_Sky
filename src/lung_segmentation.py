import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import sys

# Check if the script received the folder path as an argument
if len(sys.argv) < 5:
    print("Usage: python script_name.py <folder_path> <slice_index> <number_of_seed_points> <output_path>")
    sys.exit(1)

# Path to the original and output files
folder_path = sys.argv[1]
output_path = sys.argv[4]

# Convert slice_index to integer
try:
    slice_idx = int(sys.argv[2])
    if slice_idx < 0 or slice_idx > 9:
        print("Error: slice_index must be between 0 and 9.")
        sys.exit(1)
except ValueError:
    print("Error: slice_index must be an integer.")
    sys.exit(1)

# Convert nb_seed to integer
try:
    nb_seed = int(sys.argv[3])
    if slice_idx < 2:
        print("Error: number_of_seed_points must be larger than 2.")
        sys.exit(1)
except ValueError:
    print("Error: number_of_seed_points must be an integer.")
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

    # Disconnect after nb_seed points are selected
    if count == nb_seed:
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

segmented_image = confidence_connected
print("Segmentation pixel values:", np.unique(sitk.GetArrayFromImage(segmented_image)))
# Convert the segmented SimpleITK image to a NumPy array
segmented_np = sitk.GetArrayFromImage(segmented_image)

# Rescale the pixel values: 1 -> 255, 0 stays as 0
segmented_np_rescaled = (segmented_np * 255).astype(np.uint8)

# Convert the rescaled NumPy array back to a SimpleITK image
segmented_image_rescaled = sitk.GetImageFromArray(segmented_np_rescaled)

# Save the rescaled segmented image
output_path = f"{output_path}.nii"
sitk.WriteImage(segmented_image_rescaled, output_path)
