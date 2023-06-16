import cv2
import numpy as np


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Equalizes the histogram of the given color image. The input image should be in RGB
    color-space. The output image will be in RGB color-space as well.
    """
    # Convert from RGB color-space to YCrCb
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Equalize the histogram of the Y channel
    ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])

    # Convert back to RGB color-space from YCrCb
    equalized_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)

    return equalized_image
