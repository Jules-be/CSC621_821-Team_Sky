import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys

# Check if the script received the image path as an argument
if len(sys.argv) < 2:
    print("Usage: python histogram.py <source_image_path>")
    sys.exit(1)

# Path to the original image file
source_image_path = sys.argv[1]

# Read the original image
source_image = sitk.ReadImage(source_image_path)

# Convert the image to a numpy array
source_np = sitk.GetArrayFromImage(source_image)

# Flatten the array for histogram plotting, filter out zero values if necessary
flat_source = source_np.flatten()
flat_source = flat_source[flat_source != 0]

print("Min and max values in source image:", np.min(flat_source), np.max(flat_source))

# Check if there is data to plot
if flat_source.size > 0:
    hist_range = (np.min(flat_source), -10)
    plt.hist(flat_source, bins=256, range=hist_range)
    plt.title('Histogram of Intensity Values in Lung Region')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("No data to display in the histogram.")
