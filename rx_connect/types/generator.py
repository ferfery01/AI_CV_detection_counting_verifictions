from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as A
import numpy as np

from rx_connect.dataset_generator.composition import generate_image
from rx_connect.dataset_generator.io_utils import (
    PillMaskPaths,
    get_background_image,
    load_pill_mask_paths,
    load_random_pills_and_masks,
)
from rx_connect.tools.timers import timer


@dataclass
class RxImageGenerator:
    """The RxImageGenerator class is used to generate images of pills on a background.

    NOTE: The `images_dir` should contain the following structure:
    ├── images
    │     ├── 0001.jpg
    │     ├── ...
    ├── masks
          ├── 0001.jpg
          ├── ...

    """

    images_dir: Union[str, Path] = "RxConnectShared/ePillID/pills/"
    """Directory containing the pill images and masks. It can either be a local directory
    or remote directory.
    """
    bg_dir: Optional[Union[str, Path]] = None
    """Directory containing the background images. If None, the background images
    are randonly generated. Can also be a path to a single background image.
    """
    image_size: Tuple[int, int] = (2160, 3840)
    """The size of the generated images (height, width)
    """
    num_pills: Tuple[int, int] = (10, 50)
    """The number of pills to generate per image (min, max)
    """
    num_pills_type: int = 1
    """Different types of pills to generate
    """
    max_overlap: float = 0.2
    """Maximum overlap allowed between pills
    """
    max_attempts: int = 5
    """Maximum number of attempts to compose a pill on the background
    """
    thresh: int = 25
    """The threshold at which to binarize the mask.
    """
    color_tint: int = 0
    """Controls the aggressiveness of the color tint applied to the background.
    The higher the value, the more aggressive the color tint. The value
    should be between 0 and 10. Only used if `bg_dir` is None.
    """
    noise_var: Tuple[int, int] = (0, 100)
    """Controls the variance of the gaussian noise applied to the image.
    """
    _bg_dir: Path = field(init=False, repr=False)
    """Placeholder for the background directory
    """
    _image_dir: Path = field(init=False, repr=False)
    """Placeholder for the pill images and masks directory
    """
    _sampled_pills_path: List[Path] = field(init=False, repr=False)
    """Placeholder for the sampled pill images paths
    """
    _pill_mask_paths: PillMaskPaths = field(init=False, repr=False)
    """Placeholder for the pill images and masks paths
    """
    _bg_image: np.ndarray = field(init=False, repr=False)
    """Placeholder for the background image
    """
    _pill_images: List[np.ndarray] = field(init=False, repr=False)
    """Placeholder for the pill images
    """
    _pill_masks: List[np.ndarray] = field(init=False, repr=False)
    """Placeholder for the pill masks
    """

    def __post_init__(self) -> None:
        # Init some class attributes
        self.transform_fn = A.transforms.GaussNoise(
            var_limit=self.noise_var, always_apply=True, per_channel=True
        )

        # Load the pill images and the associated mask paths.
        self._pill_mask_paths = load_pill_mask_paths(self.images_dir)

        # Set the background image
        self.config_background(self.bg_dir, self.image_size, color_tint=self.color_tint)

        # Load the pill images and masks
        self.config_pills()

    @property
    def reference_pills(self) -> List[np.ndarray]:
        """Return the reference pill images."""
        return self._pill_images

    def config_background(
        self,
        path: Optional[Union[str, Path]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        color_tint: Optional[int] = None,
    ) -> None:
        """Configure the background image. If no path is given, a random image from the data directory is
            loaded. If no image size is given, the default image size is used.

        This method is useful if you want to generate multiple images with the same background.

        NOTE: If path is provide, it should correspond to a background image and not a directory.
        """
        image_size = image_size or self.image_size
        color_tint = color_tint or self.color_tint
        self._bg_image = get_background_image(path, *image_size, color_tint=color_tint)

    def config_pills(self) -> None:
        """Configure the pill images and masks."""
        self._pill_images, self._pill_masks = load_random_pills_and_masks(
            *self._pill_mask_paths,
            pill_types=self.num_pills_type,
            thresh=self.thresh,
        )

    @timer
    def generate(self, new_bg: bool = False, new_pill: bool = False) -> np.ndarray:
        """Generate a image with pills composed on a background image.

        Args:
            new_bg: If True, new background image is loaded. Otherwise, the loaded background
                image is used.
            new_pill: If True, new pill images and masks are loaded. Otherwise, the loaded pill
                images and masks are used.

        Returns:
            The generated synthetic image.
        """
        if new_bg:
            self.config_background(self.bg_dir)

        if new_pill:
            self.config_pills()

        img_comp, *_ = generate_image(
            self._bg_image,
            self._pill_images,
            self._pill_masks,
            min_pills=self.num_pills[0],
            max_pills=self.num_pills[1],
            max_overlap=self.max_overlap,
            max_attempts=self.max_attempts,
            enable_edge_pills=False,
        )

        return self.transform_fn(image=img_comp)["image"]
