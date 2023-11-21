import SimpleITK as sitk

# Read the images
fixed_image = sitk.ReadImage('data/covid_negative_data/patient_1/1-33.dcm', sitk.sitkFloat32)
moving_image = sitk.ReadImage('data/covid_negative_data/patient_2/1-41.dcm', sitk.sitkFloat32)

# # Turn them into 2D images
fixed_image = fixed_image[:,:,0]
moving_image = moving_image[:,:,0]

# Set up Euler Transform
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler2DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.            
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.SetInitialTransform(initial_transform, inPlace=False)
final_transform = registration_method.Execute(fixed_image, moving_image)

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

fixed_image = sitk.Cast(fixed_image, sitk.sitkInt16)
moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
# Write the images
sitk.WriteImage(fixed_image, 'data/results/fixed_image.dcm')
sitk.WriteImage(moving_image, 'data/results/moving_image.dcm')