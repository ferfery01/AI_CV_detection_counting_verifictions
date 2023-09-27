from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import albumentations as A
import numpy as np

from rx_connect import SHARED_RXIMAGE_DATA_DIR
from rx_connect.generator.composition import (
    ImageComposition,
    densify_groundtruth,
    generate_image,
)
from rx_connect.generator.io_utils import (
    enrich_metadata_with_paths,
    get_background_image,
    load_metadata,
    load_pill_mask_paths,
    load_pills_and_masks,
)
from rx_connect.generator.metadata import Colors, PillMetadata, Shapes
from rx_connect.generator.metadata_filter import (
    filter_by_color_and_shape,
    parse_file_hash,
)
from rx_connect.generator.sampler import (
    SAMPLING_METHODS_MAP,
    sample_from_color_shape_by_ndc,
)
from rx_connect.generator.transform import COMBINED_TRANSFORMS
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.timers import timer

logger = setup_logger()


@dataclass
class RxImageGenerator:
    """The RxImageGenerator class is used to generate images of pills on a background.

    NOTE: The `data_dir` should contain the following structure:
    ├── metadata.csv
    ├── images
    │     ├── 0001.jpg
    │     ├── ...
    ├── masks
          ├── 0001.png
          ├── ...
    """

    data_dir: Union[str, Path] = SHARED_RXIMAGE_DATA_DIR
    """Directory containing pill images, masks, and/or csv file containing the metadata. It can either
    be a local directory or remote directory.
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
    fraction_pills_type: Optional[Sequence[float]] = None
    """The fraction of pills per type to generate. If None, then the number of pills per type is
    randomly sampled from the range (1, num_pills_type). If a sequence is provided, then it is
    ensured that each type of pill has a specific fraction of pills in the generated image.
    """
    colors: Optional[Union[str, Sequence[str], Colors, Sequence[Colors]]] = None
    """Pills of a specific colors to generate. If None, then pills of any colors are generated.
    """
    shapes: Optional[Union[str, Sequence[str], Shapes, Sequence[Shapes]]] = None
    """Pills of a specific shapes to generate. If None, then pills of any shapes are generated.
    """
    sampling_type: str = "uniform"
    """Sampling method to use for sampling the pill images. The available methods are: uniform,
    random, and hard.
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
    apply_bg_aug: bool = True
    """Whether to apply augmentations to the background image. This is useful for generating
    images with diverse backgrounds.
    """
    apply_composed_aug: bool = True
    """Whether to apply augmentations to the composed image. This is useful for simulating
    different lighting conditions and camera conditions.
    """
    enable_defective_pills: bool = False
    """Whether to allow defective pills to be generated.
    """
    enable_edge_pills: bool = False
    """Whether to allow pills to be placed at the edge of the image.
    """

    def __post_init__(self) -> None:
        # Validate the arguments
        self._validate_args()

        # Load all the pill images and the associated masks available in the data directory
        self.data_dir = Path(self.data_dir)
        file_hash_pill_mask_paths = load_pill_mask_paths(self.data_dir)

        # Load the metadata dataframe
        _metadata_df = load_metadata(self.data_dir)
        self._metadata_df = enrich_metadata_with_paths(_metadata_df, file_hash_pill_mask_paths)

        # Filter all the pill images and masks based on the color and shape.
        self._filtered_metadata_df = filter_by_color_and_shape(
            self._metadata_df, colors=self.colors, shapes=self.shapes
        )

        # Set the background image
        self.config_background(self.bg_dir, self.image_size)

        # Load the pill images and masks
        self.config_pills()

    def _validate_args(self):
        if self.num_pills_type < 1:
            raise ValueError(
                f"`num_pills_type` should be a positive integer, but provided {self.num_pills_type}."
            )

        if self.fraction_pills_type is not None and len(self.fraction_pills_type) != self.num_pills_type:
            raise ValueError(
                f"Length of `fraction_pills_type` should be {self.num_pills_type}, but got "
                f"{len(self.fraction_pills_type)}. Please provide a fraction for each pill type."
            )

        if self.sampling_type not in SAMPLING_METHODS_MAP:
            raise ValueError(
                f"`sampling_type` should be one of {list(SAMPLING_METHODS_MAP.keys())}, but "
                f"provided {self.sampling_type}."
            )

        if self.max_overlap < 0 or self.max_overlap > 1:
            raise ValueError(f"`max_overlap` should be between 0 and 1, but provided {self.max_overlap}.")

        if self.max_attempts <= 0:
            raise ValueError(
                f"`max_attempts` should be a positive integer, but provided {self.max_attempts}."
            )

    @property
    def sampled_images_path(self) -> List[Path]:
        """Return the sampled pill images paths."""
        return [pill_mask_path.image_path for pill_mask_path in self._sampled_pill_mask_paths]

    @property
    def reference_pills(self) -> List[np.ndarray]:
        """Return the reference pill images."""
        densified_references = []
        for i, pill_image in enumerate(self._pill_images):
            xmin, xmax, ymin, ymax = densify_groundtruth(self._pill_masks[i])
            densified_references.append(pill_image[xmin:xmax, ymin:ymax])
        return densified_references

    @property
    def metadata(self) -> List[PillMetadata]:
        """Returns the metadata for the sampled reference pills. The order of the metadata is the same as the
        order of the sampled pills. If a particular metadata is not available, then the corresponding
        field is set to None.
        """
        metadata_list: List[PillMetadata] = []
        for img_path in self.sampled_images_path:
            file_hash = parse_file_hash(img_path)
            metadata = self._metadata_df.loc[file_hash].to_dict()
            metadata_list.append(
                PillMetadata(
                    drug_name=metadata.get("GenericName"),
                    ndc=metadata.get("NDC9"),
                    color=metadata.get("Color"),
                    shape=metadata.get("Shape"),
                    imprint=metadata.get("Imprint"),
                )
            )
        return metadata_list

    def config_background(
        self, path: Optional[Union[str, Path]] = None, image_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """Configure the background image. If no path is given, a random colored background is loaded.
        If a directory is provided, a random image is selected in each call.
        If no image size is given, the default image size is used.

        Calling this method will reset the background image to the new image. You can also adjust the
        image size by passing the `image_size` during the runtime.
        """
        image_size = image_size or self.image_size
        self._bg_image = get_background_image(path, *image_size, apply_augmentations=self.apply_bg_aug)

    def config_pills(self) -> None:
        """Configure the pill images and masks."""
        self._sampled_pill_mask_paths = sample_from_color_shape_by_ndc(
            self._filtered_metadata_df, pill_types=self.num_pills_type, sampling=self.sampling_type
        )
        self._pill_images, self._pill_masks = load_pills_and_masks(
            self._sampled_pill_mask_paths, thresh=self.thresh
        )

    def __call__(self, new_bg: bool = True, new_pill: bool = True) -> ImageComposition:
        """Generate synthetic images along with the masks and labels.

        Args:
            new_bg: If True, new background image is loaded, otherwise, the last loaded background
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

        # Generate the composed image
        img_comp, mask_comp, labels_comp, gt_bbox, pills_per_type = generate_image(
            self._bg_image,
            self._pill_images,
            self._pill_masks,
            min_pills=self.num_pills[0],
            max_pills=self.num_pills[1],
            fraction_pills_type=self.fraction_pills_type,
            scale=self.scale,
            max_overlap=self.max_overlap,
            max_attempts=self.max_attempts,
            enable_defective_pills=self.enable_defective_pills,
            enable_edge_pills=self.enable_edge_pills,
        )

        # Apply augmentations to the composed image
        if self.apply_composed_aug:
            transform = A.OneOf(COMBINED_TRANSFORMS)
            img_comp = transform(image=img_comp)["image"]

        return ImageComposition(img_comp, mask_comp, labels_comp, gt_bbox, pills_per_type)

    @timer()
    def generate(self, new_bg: bool = True, new_pill: bool = True) -> ImageComposition:
        """This is an alias for __call__."""
        return self(new_bg, new_pill)
