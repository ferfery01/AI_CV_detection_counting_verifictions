from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import av
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video, clear_output, display
from PIL import Image
from tqdm import tqdm

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".avi", ".mov", ".wmv", ".flv", ".gif")


class VideoCodec(Enum):
    """Enum class representing various video codecs.

    Each codec is represented by a four character code. The availability
    of codecs depends on the system and the installed libraries.
    """

    MPEG4 = "mpeg4"
    """MPEG-4 part 2. Good balance of file size and quality, but may not offer as high quality as
    more recent codecs. Works with .mp4 files.
    """
    H264 = "h264"
    """H.264, also known as AVC (Advanced Video Coding). Very common and widely supported codec,
    offers good quality and file size. Works with .mp4 files.
    """
    HEVC = "hevc"
    """H.265, also known as HEVC (High Efficiency Video Coding). More efficient than H.264 in terms
    of quality per file size, but requires more processing power to encode and decode. Works with
    .mp4 files.
    """
    VP9 = "vp9"
    """A codec developed by Google as a successor to VP8. Used in YouTube and WebM video. Works with
    .webm files.
    """
    AV1 = "av1"
    """A successor to VP9, developed by the Alliance for Open Media. More efficient than VP9 or H.264,
    but requires significantly more processing power to encode. Works with .webm files.
    """
    FLV1 = "flv1"
    """Flash Video codec. Generally not recommended due to the obsolescence of Flash. Works with .flv
    files.
    """
    wmv2 = "wmv2"
    """Windows Media Video codec. Generally not recommended due to the obsolescence of Windows Media
    Player. Works with .wmv files.
    """


def load_video_frame(
    path: Union[str, Path], n_frames: Optional[int] = None, target_fps: Optional[int] = None
) -> Iterator[np.ndarray]:
    """This function opens a video file and yields its frames one at a time. It allows for
    optional control over the number of frames loaded and the target frames per second (FPS).

    Args:
        path (Union[str, Path]): Path to the video file.
        n_frames (Optional[int], optional): Number of frames to load. Defaults to None.
            If None, all the frames are loaded.
        target_fps (Optional[int], optional): Target FPS of the video. Defaults to None.
            If None, the original FPS of the video is used.

    Yields:
        Iterator[np.ndarray]: An iterator that yields frames of the video in RGB format. Each
        frame is a numpy ndarray with shape (height, width, 3).

    Raises:
        AssertionError: If the video file specified by path does not exist.
    """
    assert Path(path).exists(), f"Couldn't find video file at {path}."

    # Open the video file
    with av.open(path) as container:
        # Get the video stream
        stream = next(s for s in container.streams if s.type == "video")

        # Get the total number of frames in the video
        total_frames = stream.frames

        # If the number of frames is not specified, load all the frames
        n_frames = total_frames if n_frames is None else min(n_frames, total_frames)

        # Get the original video's FPS
        original_fps = stream.average_rate

        # Calculate frame step based on target FPS
        frame_step = 1
        if target_fps is not None and original_fps > target_fps:
            frame_step = int(original_fps / target_fps)

        # Calculate the actual number of frames that will be loaded
        actual_num_frames = min(n_frames, total_frames // frame_step)

        # Allow multiple theads to decode independent frames
        container.streams.video[0].thread_type = "AUTO"

        with tqdm(total=actual_num_frames, desc="Loading frames", unit="frame") as pbar:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    # We skip frames here based on frame_step
                    if frame.index % frame_step != 0:
                        continue

                    # Convert the frame to RGB format and yield it
                    yield frame.to_ndarray(format="rgb24")
                    pbar.update(1)

                    if frame.index >= n_frames:
                        break


def load_video(
    path: Union[str, Path], n_frames: Optional[int] = None, target_fps: Optional[int] = None
) -> List[np.ndarray]:
    """This function opens a video file and loads all its frames into memory. See `load_video_frame`
    for more details on the arguments.
    """
    return list(load_video_frame(path, n_frames, target_fps))


def save_gif(frames: List[np.ndarray], path: Union[str, Path], fps: int) -> None:
    """Save a video locally in .gif format.

    Args:
        frames (List[np.ndarray]): Frames representing the video, each in (H, W, C), RGB. Note
            that all the frames are expected to have the same shape.
        path (Union[str, Path]): Where the video will be saved
        fps (int): Frames per second
    """
    # Convert the frames to PIL images
    frames_pil = [Image.fromarray(frame) for frame in frames]

    # Save the frames as a GIF
    frames_pil[0].save(path, save_all=True, append_images=frames_pil[1:], duration=int(1000 / fps), loop=0)

    # Log the path to the saved GIF
    logger.info(f"Saved gif to {path}")


def save_mp4(frames: List[np.ndarray], path: Union[str, Path], fps: int, codec: str = "mpeg4") -> None:
    """Save a video locally using av library.

    Args:
        frames (List[np.ndarray]): Frames representing the video, each in (H, W, C), RGB format.
        path (Union[str, Path]): Where the video will be saved
        fps (int): Frames per second
        codec (str): Codec to use for the video. Defaults to "mpeg4".
    """
    path = str(path) if isinstance(path, Path) else path
    height, width, _ = frames[0].shape

    with av.open(path, mode="w") as container:
        # Create a video stream in the output container with specified codec
        codec = VideoCodec(codec).value
        stream = container.add_stream(codec, rate=fps)
        stream.height = height
        stream.width = width

        for frame in tqdm(frames, desc="Writing video", unit="frame"):
            # Create a new AVFrame from the numpy array
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

            # Encode and write the frame to the container
            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush any remaining packets to the container
        for packet in stream.encode():
            container.mux(packet)

    # Log the path to the saved video
    logger.info(f"Saved video to {path}")


def save_video(frames: List[np.ndarray], path: Union[str, Path], fps: int, codec: str = "mpeg4") -> None:
    """Save a video locally. Depending on the extension, the video will be saved as a .mp4 file
    or as a .gif file.

    Args:
        frames (List[np.ndarray]): Frames representing the video, each in (H, W, C), RGB. Note
            that all the frames are expected to have the same shape.
        path (Union[str, Path]): Where the video will be saved
        fps (int): Frames per second
        codec (str): Codec to use for the video.
    """
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        logger.info(
            f'Output path "{path}" does not have a video extension, and therefore will be saved as {path}.mp4'
        )
        path = path.with_suffix(".mp4")

    if path.suffix.lower() == ".gif":
        save_gif(frames, path, fps)
    else:
        save_mp4(frames, path, fps, codec)


def _is_list_of_ndarrays(lst: List[np.ndarray]) -> bool:
    """Checks if lst is a list of numpy arrays."""
    if isinstance(lst, list):
        # checks if all elements in lst are numpy arrays
        return all(isinstance(i, np.ndarray) for i in lst)
    return False


def display_video(video: Union[str, Path, List[np.ndarray]]) -> None:
    """Display a video in a Jupyter notebook. The video can be specified either as a path to a video
    file or as a list of frames.
    """
    # Display the video in the notebook
    if isinstance(video, (str, Path)):
        display(Video(video))
    elif _is_list_of_ndarrays(video):
        for frame in video:
            plt.imshow(frame)
            plt.axis("off")
            plt.show()
            clear_output(wait=True)
    else:
        raise TypeError(f"Invalid type for video: {type(video)}")
