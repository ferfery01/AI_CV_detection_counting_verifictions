import cv2
import numpy as np


def generate_grayscale_mask(image: np.ndarray, thresh: int = 0) -> np.ndarray:
    """Generates a boolean mask for the image."""
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a boolean mask where the grayscale image is greater than threshold
    boolean_mask = gray_image > thresh

    return boolean_mask


def fill_largest_contour(mask: np.ndarray, fill_value: int = 1) -> np.ndarray:
    """Fills the largest contour in the mask with the specified value and set
    anything outside the contour to 0.
    """
    mask = mask.astype(np.uint8).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)

    # Check if there are any contours
    if contours:
        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Draw the contours with a specific value
        cv2.drawContours(mask, contours, 0, fill_value, thickness=cv2.FILLED)

    return mask > 0
