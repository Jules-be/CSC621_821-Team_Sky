
import SimpleITK as sitk
import sys
import os

reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames("data/covid_negative_data/patient_1")
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print("Image size:", size[0], size[1], size[2])

print("Writing image:", "data/results/patient_1.mha")

sitk.WriteImage(image, "data/results/patient_1.mha")
