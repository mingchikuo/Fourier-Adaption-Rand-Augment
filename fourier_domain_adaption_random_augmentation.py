import cv2
import numpy as np

def fourier_domain_adaption(source_image_path, target_image_path, cut_length=(32, 8)):
    """
    Merge the low-frequency component of the source image and the high-frequency component of the target image
    to create a hybrid image using Fourier Transform.

    Parameters:
    source_image_path (str): The file path of the source image.
    target_image_path (str): The file path of the target image.
    cut_length (tuple): A tuple containing two elements, the first element is the center of the range of the possible
        values of the cut length, and the second element is the range of the possible values of the cut length.
        The cut length is randomly selected from this range and then the actual cut length is calculated by adding a
        normally-distributed random value with mean 0 and standard deviation 1/4 of the range.

    Returns:
    merged_image (ndarray): The transformed image.
    """
    
    # Load source and target images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)

    # Split the source and target images into their color channels
    source_bgr = cv2.split(source_image)
    target_bgr = cv2.split(target_image)

    transformed_bgr = []
    for i in range(3):
        # Convert the current channel of source and target images to grayscale
        source_gray = cv2.cvtColor(source_bgr[i], cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_bgr[i], cv2.COLOR_BGR2GRAY)

        # Apply Fourier Transform to the current channel of source and target images
        source_dft = cv2.dft(np.float32(source_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        target_dft = cv2.dft(np.float32(target_gray), flags=cv2.DFT_COMPLEX_OUTPUT)

        # Merge the low-frequency component of the current channel of target image 
        # and   the high-frequency component of the current channel of source image 
        # by creating a mask
        rows, cols = source_gray.shape
        crow, ccol = rows // 2, cols // 2
        mean, range_ = cut_length
        cut_length = mean + int(np.random.uniform(-range_ / 2, range_ / 2))
        cut_length += int(np.random.normal(0, range_ / 4))
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow-cut_length:crow+cut_length, ccol-cut_length:ccol+cut_length] = 1
        fshift1 = source_dft * mask
        fshift2 = target_dft * (1 - mask)
        merged_dft = fshift1 + fshift2

        # Apply Inverse Fourier Transform to merged image
        merged_idft = cv2.idft(merged_dft)
        merged_idft = cv2.magnitude(merged_idft[:, :, 0], merged_idft[:, :, 1])

        # Normalize the merged image to the range of 0-255
        cv2.normalize(merged_idft, merged_idft, 0, 255, cv2.NORM_MINMAX)

        # Convert the merged image to 8-bit unsigned integer
        transformed_channel = np.uint8(merged_idft)

        # Append the transformed channel to the list
        transformed_bgr.append(transformed_channel)

    # Merge the transformed channels into a single image
    transformed_image = cv2.merge(transformed_bgr)

    return transformed_image
