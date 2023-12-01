#!/usr/bin/env python

import SimpleITK as sitk
import sys
import os


def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")


if len(sys.argv) < 4:
    print(
        f"Usage: {sys.argv[0]}"
        + " <fixedImageFilter> <movingImageFile> <outputTransformFile>"
    )
    sys.exit(1)

#fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)


fixed = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(sys.argv[1]))
moving = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(sys.argv[2]))

#moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)


resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
moving = resampler.Execute(moving)


matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving, fixed)

# The basic Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in
# SimpleITK
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(50)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)

demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

displacementField = demons.Execute(fixed, moving)

print("-------")
print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
print(f" RMS: {demons.GetRMSChange()}")

output_path = r'C:\Users\feras\Desktop\CSC621_821-Team_Sky\output_image.nii'
sitk.WriteImage(displacementField, output_path)
outTx = sitk.DisplacementFieldTransform(displacementField)

sitk.WriteTransform( outTx , sys.argv[3] + ".h5")

if "SITK_NOSHOW" not in os.environ:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # Use the // floor division operator so that the pixel type is
    # the same for all three images which is the expectation for
    # the compose filter.
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    sitk.Show(cimg, "DeformableRegistration1 Composition")