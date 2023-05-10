"""example script for diffeomorphic demons registration using circle to C problem
   based on
   
   https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/66_Registration_Demons.ipynb
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors] * image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas] * image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(sz / float(sf) + 0.5)
        for sf, sz in zip(shrink_factors, original_size)
    ]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                   for original_sz, original_spc, new_sz in zip(
                       original_size, original_spacing, new_size)]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    shrink_factors=None,
    smoothing_sigmas=None,
):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry,
                                                   shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                   When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                   This allows us to specify different shrink factors per axis. This is useful
                                                   in the context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we would only want to
                                                   sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """

    # Create image pyramid in a memory efficient manner using a generator function.
    # The whole pyramid never exists in memory, each level is created when iterating over
    # the generator.
    def image_pair_generator(fixed_image, moving_image, shrink_factors,
                             smoothing_sigmas):
        end_level = 0
        start_level = 0
        if shrink_factors is not None:
            end_level = len(shrink_factors)
        for level in range(start_level, end_level):
            f_image = smooth_and_resample(fixed_image, shrink_factors[level],
                                          smoothing_sigmas[level])
            m_image = smooth_and_resample(moving_image, shrink_factors[level],
                                          smoothing_sigmas[level])
            yield (f_image, m_image)
        yield (fixed_image, moving_image)

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because
    # of a constraint imposed by the Demons filters.
    if shrink_factors is not None:
        original_size = fixed_image.GetSize()
        original_spacing = fixed_image.GetSpacing()
        s_factors = ([shrink_factors[0]] * len(original_size)
                     if np.isscalar(shrink_factors[0]) else shrink_factors[0])
        df_size = [
            int(sz / float(sf) + 0.5)
            for sf, sz in zip(s_factors, original_size)
        ]
        df_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                      for original_sz, original_spc, new_sz in zip(
                          original_size, original_spacing, df_size)]
    else:
        df_size = fixed_image.GetSize()
        df_spacing = fixed_image.GetSpacing()

    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(
            initial_transform,
            sitk.sitkVectorFloat64,
            df_size,
            fixed_image.GetOrigin(),
            df_spacing,
            fixed_image.GetDirection(),
        )
    else:
        initial_displacement_field = sitk.Image(df_size,
                                                sitk.sitkVectorFloat64,
                                                fixed_image.GetDimension())
        initial_displacement_field.SetSpacing(df_spacing)
        initial_displacement_field.SetOrigin(fixed_image.GetOrigin())

    # Run the registration.
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in image_pair_generator(fixed_image, moving_image,
                                                 shrink_factors,
                                                 smoothing_sigmas):
        initial_displacement_field = sitk.Resample(initial_displacement_field,
                                                   f_image)
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


def resample(image, transform, default_value=0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkLinear
    return sitk.Resample(image, reference_image, transform, interpolator,
                         default_value)


#################################################################################################
#################################################################################################
#################################################################################################

if __name__ == "__main__":
    # generate images
    n = 256
    r = 0.7
    noise_level = 0.075

    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    fixed_np = np.zeros((n, n))
    moving_np = np.zeros((n, n))

    moving_np[R <= r] = 1

    w = n // 15

    fixed_np[R <= r] = 1
    fixed_np[R <= 0.5 * r] = 0
    fixed_np[((n // 2) - w):((n // 2) + w), (n // 2):] = 0

    fixed_np += noise_level * np.random.randn(n, n)
    moving_np += noise_level * np.random.randn(n, n)

    fixed = sitk.GetImageFromArray(fixed_np)
    moving = sitk.GetImageFromArray(moving_np)

    # Select a Demons filter and configure it.
    #demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(1500)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(0.75)

    # Run the registration.
    tx = multiscale_demons(
        registration_algorithm=demons_filter,
        fixed_image=fixed,
        moving_image=moving,
        shrink_factors=[4, 2],
        smoothing_sigmas=[8, 4],  #[8,4]
    )

    # resample the image
    moving_resampled = resample(moving, tx)

    # resample grid image
    grid = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                           size=(n, n),
                           sigma=(0.1, 0.1),
                           gridSpacing=(5.0, 5.0))
    grid_resampled = resample(grid, tx)

    # Invert a displacement field transform
    inverse_tx = sitk.DisplacementFieldTransform(
        sitk.InvertDisplacementField(
            tx.GetDisplacementField(),
            maximumNumberOfIterations=1000,
            maxErrorToleranceThreshold=0.001,
            meanErrorToleranceThreshold=0.00001,
            enforceBoundaryCondition=True,
        ))

    tmp1 = resample(moving_resampled, inverse_tx)
    tmp2 = resample(grid_resampled, inverse_tx)

    jac = sitk.DisplacementFieldJacobianDeterminantFilter().Execute(
        tx.GetDisplacementField())

# show results
ims = dict(vmin=0, vmax=1.1)
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
ax[0, 0].imshow(sitk.GetArrayFromImage(fixed), **ims)
ax[0, 1].imshow(sitk.GetArrayFromImage(moving), **ims)
ax[0, 2].imshow(sitk.GetArrayFromImage(moving_resampled), **ims)
ax[0, 3].imshow(sitk.GetArrayFromImage(tmp1), **ims)
ax[1, 0].imshow(sitk.GetArrayFromImage(grid))
ax[1, 1].imshow(sitk.GetArrayFromImage(grid_resampled))
ax[1, 2].imshow(sitk.GetArrayFromImage(tmp2))
ax[1, 3].imshow(sitk.GetArrayFromImage(jac))

for axx in ax.ravel():
    axx.set_axis_off()

ax[0, 0].set_title('fixed')
ax[0, 1].set_title('moving')
ax[0, 2].set_title('S(moving)')
ax[0, 3].set_title('S_inv(S(moving))')
ax[1, 0].set_title('grid')
ax[1, 1].set_title('S(grid)')
ax[1, 2].set_title('S_inv(S(grid))')
ax[1, 3].set_title('S(Jacobian)')

fig.tight_layout()
fig.show()
