import logging
from typing import Any, Dict, List, Tuple, cast

import gradio as gr
import numpy as np

from rx_connect import CACHE_DIR
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.generator import RxImageGenerator
from rx_connect.pipelines.image import RxVision
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizerColorhist
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.timers import timer

logger = setup_logger()
log_file = CACHE_DIR / "demo_v2.log"
logger.addHandler(logging.FileHandler(log_file))

IMAGE_OBJ = RxVision()
STREAM_OBJ = RxVision()
GENERATOR_OBJ = None
CURRENT_SOURCE = "upload"


def demo_init() -> None:
    """
    Demo event function - run initialization steps:
        1. Tool objects: GENERATOR_OBJ, counterObj, segmentObj, vectorizerObj
        2. Image pipeline
    """
    open(log_file, "w").close()
    logger.info("Initialization started...")
    with timer() as t:
        global GENERATOR_OBJ

        logger.info("\tPreparing image generation tool...")
        GENERATOR_OBJ = RxImageGenerator(num_pills_type=2)

        logger.info("\tPreparing pill detection tool...")
        counterObj = RxDetection()

        logger.info("\tPreparing segmentation tool...")
        segmentObj = RxSegmentation()

        logger.info("\tPreparing vectorization tool...")
        vectorizerObj = RxVectorizerColorhist()

        logger.info("\tPreparing image pipeline...")
        IMAGE_OBJ.set_counter(counterObj)
        IMAGE_OBJ.set_segmenter(segmentObj)
        IMAGE_OBJ.set_vectorizer(vectorizerObj)
        STREAM_OBJ.set_counter(counterObj)

    logger.info(f"...Initialization finished. Spent {t.duration:.2f} seconds.")


def demo_choose_source() -> Dict[str, Any]:
    """
    Demo event function - selects image source upload/camera.
    """
    global CURRENT_SOURCE
    CURRENT_SOURCE = "webcam" if CURRENT_SOURCE == "upload" else "upload"
    logger.info(f"Swithing image source to {CURRENT_SOURCE}.")
    return {
        "value": None,
        "source": CURRENT_SOURCE,
        "streaming": False,
        "__type__": "update",
    }


def demo_load_image(image: np.ndarray, ref_image: np.ndarray, MSG: str) -> None:
    """
    Demo event function - load image from Image containers.
    """
    IMAGE_OBJ.load_image(image)
    IMAGE_OBJ.load_ref_image(ref_image)
    logger.info(f"Input updated: {MSG}.")


def demo_gen_image() -> Tuple[np.ndarray, np.ndarray]:
    """
    Demo event function - generate and load images.
    """
    logger.info("Generation started...")
    with timer() as t:
        IMAGE_OBJ.load_from_generator(cast(RxImageGenerator, GENERATOR_OBJ))
    logger.info(f"...Generation finished. Spent {t.duration:.2f} seconds.")
    return IMAGE_OBJ.image, IMAGE_OBJ.ref_image


def demo_detect() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Demo event function - detect trigger button.
    """
    logger.info("Detection started...")
    with timer() as t:
        ROI, BB = IMAGE_OBJ.ROIs, IMAGE_OBJ.draw_bounding_boxes()
    logger.info(f"...Detection finished. Spent {t.duration:.2f} seconds.")
    return ROI, BB


def demo_segment() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Demo event function - segment trigger button.
    """
    logger.info("Segmentation started...")
    with timer() as t:
        mROI, mFull = IMAGE_OBJ.masked_ROIs, IMAGE_OBJ.background_segment
    logger.info(f"...Segmentation finished. Spent {t.duration:.2f} seconds.")
    return mROI, mFull * 255


def demo_verify() -> list[tuple[np.ndarray, str]]:
    """
    Demo event function - verify trigger button.
    """
    logger.info("Verification started...")
    with timer() as t:
        verify_scores = [
            (ROI, f"{score:.3f}") for ROI, score in zip(IMAGE_OBJ.ROIs, IMAGE_OBJ.similarity_scores)
        ]
    logger.info(f"...Verification finished. Spent {t.duration:.2f} seconds.")
    return verify_scores


def demo_start_streaming() -> gr.update:
    logger.info("Starting Camera...")
    return {
        "value": None,
        "source": "webcam",
        "interactive": True,
        "streaming": True,
        "mirror_webcam": False,
        "__type__": "update",
    }


def demo_stop_streaming(current_frame: np.ndarray) -> Tuple[gr.update, np.ndarray]:
    logger.info("Stopping Camera...")
    return (
        {
            "value": None,
            "source": "upload",
            "interactive": False,
            "streaming": False,
            "__type__": "update",
        },
        current_frame,
    )


def demo_video_counter(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Demo event function - video counter utility.
    """
    STREAM_OBJ.load_image(frame)
    return STREAM_OBJ.draw_bounding_boxes(), f"Current pill count: {STREAM_OBJ.pill_count}."


def demo_read_logs() -> str:
    """
    Demo event function - read log file to show.
    """
    with open(log_file, "r") as f:
        return f.read()


def create_demo() -> gr.Blocks:
    """
    Demo visualization module structure - defines the elements and the function hooks.
    """
    with gr.Blocks(css="footer {visibility: hidden}", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RxVision")
        with gr.Tab("Image Loading"):
            with gr.Row():
                image_input = gr.Image(label="Image to verify", height=480)
                image_ref = gr.Image(label="Reference image", height=480)
            with gr.Row():
                image_source_button = gr.Button("Upload / Camera", variant="primary")
                image_gen_button = gr.Button("Generate (for testing purposes)", variant="secondary")
        with gr.Tab("Detection"):
            with gr.Row():
                gallery_ROI = gr.Gallery(label="ROIs", columns=5, object_fit="scale-down", height=480)
                image_BB = gr.Image(label="Bounding boxes", height=480)
            image_detect_button = gr.Button("Detect", variant="primary")
        with gr.Tab("Segmentation"):
            with gr.Row():
                gallery_mROI = gr.Gallery(label="Masked ROIs", columns=5, object_fit="scale-down", height=480)
                image_mFull = gr.Image(label="Bachground segmentations", height=480)
            image_segment_button = gr.Button("Segment", variant="primary")
        with gr.Tab("Verification"):
            gallery_scores = gr.Gallery(
                label="Verification Scores", columns=5, object_fit="scale-down", height=480
            )
            image_verify_button = gr.Button("Verify", variant="primary")
        with gr.Tab("Counting Helper"):
            with gr.Column():
                streaming_source = gr.Image(
                    label="Input camera stream",
                    source="upload",
                    interactive=False,
                    streaming=False,
                    visible=False,
                )
                streaming_destination = gr.Image(label="Output result", height=480)
            with gr.Row():
                CountHelpBox = gr.Textbox(show_label=False)
                stream_start_button = gr.Button("Start Streaming", variant="primary")
                stream_stop_button = gr.Button("Stop Streaming and Load Image", variant="stop")

        with gr.Accordion("Logs"):
            gr.Textbox(demo_read_logs, every=1, show_label=False)

        image_source_button.click(demo_choose_source, None, image_input)
        image_input.change(
            lambda img, img_ref: demo_load_image(img, img_ref, image_input.label),
            inputs=[image_input, image_ref],
            outputs=None,
        )
        image_ref.change(
            lambda img, img_ref: demo_load_image(img, img_ref, image_ref.label),
            inputs=[image_input, image_ref],
            outputs=None,
        )
        image_gen_button.click(demo_gen_image, inputs=None, outputs=[image_input, image_ref])
        image_detect_button.click(demo_detect, None, [gallery_ROI, image_BB])
        image_segment_button.click(demo_segment, None, [gallery_mROI, image_mFull])
        image_verify_button.click(demo_verify, None, gallery_scores)
        stream_start_button.click(demo_start_streaming, None, streaming_source)
        stream_stop_button.click(demo_stop_streaming, streaming_source, [streaming_source, image_input])
        streaming_source.change(
            demo_video_counter,
            streaming_source,
            [streaming_destination, CountHelpBox],
            show_progress="hidden",
        )
        demo.load(demo_init)
        demo.queue()
    return demo


if __name__ == "__main__":
    create_demo().launch(share=True)
