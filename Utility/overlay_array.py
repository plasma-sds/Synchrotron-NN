#######################################################################
##
## Script define the function to overlay two images with one of them
## divided by arbitrary factor to reduce intensity. Noise can be added
## optionally.
##
## Created by Soma Olasz, 05.2025
##
#######################################################################

import numpy as np
from PIL import Image


def combine_images(image1, image2, beta, a_value, add_noise=False):
    # Load the images and convert them to grayscale
    image1 = image1.convert('L')  # Convert to grayscale ('L' mode)
    image2 = image2.convert('L')

    # Resize images to common size
    image1 = image1.resize(image2.size)

    # Convert images to NumPy arrays
    image1_array = np.array(image1) / beta
    image2_array = np.array(image2)

    # Add the two arrays element-wise
    result_array = (image1_array.astype(np.int16) + image2_array.astype(np.int16))

    # Clip values to ensure they stay within the valid range (0-255)
    result_array = np.clip(result_array, 0, 255)
    
    # Optional: Add signal-scaled Gaussian noise
    if add_noise:
        result_array = add_signal_scaled_noise(result_array)
     
    # Make a mask of the region where SOFT image is non-zero
    mask = (image1_array != 0).astype(np.uint8)

    # Calculate the ratio of the masked images
    masked_ratio = (result_array/image2_array) * mask
    masked_ratio[~np.isfinite(masked_ratio)] = 0
     
    # Calculate alpha to esitmate when runaway radiation can be seen on the result image
    alpha = np.median(masked_ratio[masked_ratio != 0])

    # Convert the resulting array back to an image
    result_image = Image.fromarray(result_array.astype(np.uint8))
    return result_image, alpha

def add_signal_scaled_noise(image_array):
    """
    Add Gaussian noise to the image, where the noise level scales with intensity.
    Stronger signals get ~5% noise, weaker ones get up to ~20%.
    """
    
    # Normalize image to [0, 1] range
    normalized = image_array / 255.0

    # Inverse intensity for scaling: higher noise for lower intensities
    # Scale between 0.2 (20%) for low values to 0.05 (5%) for high
    noise_scale = 0.2 - 0.15 * normalized  # Linear interpolation
    noise_std = noise_scale * image_array

    # Generate Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std)

    # Add noise and return
    noisy_image = image_array + noise
    return noisy_image
