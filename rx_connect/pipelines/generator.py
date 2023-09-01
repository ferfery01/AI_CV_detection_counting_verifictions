from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from rx_connect import SHARED_RXIMAGE_DATA_DIR
from rx_connect.core.types.generator import PillMaskPaths
from rx_connect.generator.composition import (
    ImageComposition,
    densify_groundtruth,
    generate_image,
)
from rx_connect.generator.io_utils import (
    get_background_image,
    load_pill_mask_paths,
    load_pills_and_masks,
    random_sample_pills,
)
from rx_connect.generator.transform import apply_augmentations
from rx_connect.tools.timers import timer


@dataclass
class RxImageGenerator:
    """The RxImageGenerator class is used to generate images of pills on a background.

    NOTE: The `data_dir` should contain the following structure:
    ├── images
    │     ├── 0001.jpg
    │     ├── ...
    ├── masks
          ├── 0001.jpg(png)
          ├── ...
    """

    data_dir: Union[str, Path] = SHARED_RXIMAGE_DATA_DIR
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
    scale: Union[float, Tuple[float, float]] = (0.3, 0.3)
    """The scaling factor to use for rescaling the pill image and mask. If a tuple is provided,
    then the scaling factor is randomly sampled from the range (min, max). If a float is
    provided, then the scaling factor is fixed.
    """
    max_overlap: float = 0.2
    """Maximum overlap allowed between pills. The value should be between 0 and 1. If the value
    is 0, then no overlap is allowed. If the value is 1, then the pills can overlap completely.
    """
    max_attempts: int = 5
    """Maximum number of attempts to compose a pill on the background. If the number of attempts
    exceeds this value, then the pill is discarded.
    """
    thresh: int = 25
    """The threshold at which to binarize the mask. Useful only for the old ePillID masks.
    """
    color_tint: int = 0
    """Controls the aggressiveness of the color tint applied to the background.
    The higher the value, the more aggressive the color tint. The value
    should be between 0 and 20. Only used if `bg_dir` is None.
    """
    apply_color: bool = True
    """Whether to apply color augmentations to the generated images. This is useful
    for generating images with different lighting conditions.
    """
    apply_noise: bool = True
    """Whether to apply noise augmentations to the generated images. This is useful
    for simulating images taken with different camera conditions.
    """
    enable_defective_pills: bool = False
    """Whether to allow defective pills to be generated.
    """
    enable_edge_pills: bool = False
    """Whether to allow pills to be placed at the edge of the image.
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
        # Load the pill images and the associated mask paths.
        self._pill_mask_paths = load_pill_mask_paths(self.data_dir)

        # Set the background image
        self.config_background(self.bg_dir, self.image_size, color_tint=self.color_tint)

        # Load the pill images and masks
        self.config_pills()

    @property
    def sampled_images_path(self) -> List[Path]:
        """Return the sampled pill images paths."""
        return self._sampled_pills_path

    @property
    def reference_pills(self) -> List[np.ndarray]:
        """Return the reference pill images."""
        densified_references = []
        for i, pill_image in enumerate(self._pill_images):
            xmin, xmax, ymin, ymax = densify_groundtruth(self._pill_masks[i])
            densified_references.append(pill_image[xmin:xmax, ymin:ymax])
        return densified_references

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
        self._sampled_pills_path, sampled_masks_path = random_sample_pills(
            *self._pill_mask_paths, self.num_pills_type
        )
        self._pill_images, self._pill_masks = load_pills_and_masks(
            self._sampled_pills_path, sampled_masks_path, thresh=self.thresh, color_aug=True
        )

    def __call__(self, new_bg: bool = True, new_pill: bool = True) -> ImageComposition:
        """Generate synthetic images along with the masks and labels.

        Args:
            new_bg: If True, new background image is loaded. Otherwise, the loaded background
                image is used.
            new_pill: If True, new pill images and masks are loaded. Otherwise, the loaded pill
                images and masks are used.

        Returns:
            ImageComposition: The generated result, including
            (image, mask, labels, groundtruth bounding boxes, number of pills per type).
        """
        if new_bg:
            self.config_background(self.bg_dir)

        if new_pill:
            self.config_pills()

        img_comp, mask_comp, labels_comp, gt_bbox, pills_per_type = generate_image(
            self._bg_image,
            self._pill_images,
            self._pill_masks,
            min_pills=self.num_pills[0],
            max_pills=self.num_pills[1],
            scale=self.scale,
            max_overlap=self.max_overlap,
            max_attempts=self.max_attempts,
            enable_defective_pills=self.enable_defective_pills,
            enable_edge_pills=self.enable_edge_pills,
        )

        # Apply color and/or noise augmentations
        img_comp = apply_augmentations(img_comp, apply_color=self.apply_color, apply_noise=self.apply_noise)

        return ImageComposition(img_comp, mask_comp, labels_comp, gt_bbox, pills_per_type)

    @timer()
    def generate(self, new_bg: bool = True, new_pill: bool = True) -> ImageComposition:
        """This is an alias for __call__."""
        return self(new_bg, new_pill)
