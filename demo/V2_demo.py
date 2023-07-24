import logging
import time
from typing import Any, Dict, List, Tuple, cast

import gradio as gr
import numpy as np

from rx_connect import CACHE_DIR
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.generator import RxImageGenerator
from rx_connect.pipelines.image import RxImage
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizerColorhist
from rx_connect.tools.logging import setup_logger

logger = setup_logger()
log_file = CACHE_DIR / "demo_v2.log"
logger.addHandler(logging.FileHandler(log_file))

IMAGE_OBJ = RxImage()
STREAM_OBJ = RxImage()
GENERATOR_OBJ = None


def demo_init() -> None:
    """
    Demo event function - run initialization steps:
        1. Tool objects: GENERATOR_OBJ, counterObj, segmentObj, vectorizerObj
        2. Image pipeline
    """
    open(log_file, "w").close()
    logger.info("Initialization started...")
    t = time.time()
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
    logger.info(f"...Initialization finished. Spent {time.time()-t:.2f} seconds.")


def demo_choose_source(source: str) -> Dict[str, Any]:
    """
    Demo event function - selects image source upload/camera.
    """
    return {
        "value": None,
        "source": "webcam" if source == "Camera" else "upload",
        "streaming": False,
        "__type__": "update",
    }


def demo_load_image(image: np.ndarray, ref_image: np.ndarray) -> None:
    """
    Demo event function - load image from Image containers.
    """
    IMAGE_OBJ.load_image(image)
    IMAGE_OBJ.load_ref_image(ref_image)
    logger.info("Image Loaded.")


def demo_gen_image() -> Tuple[np.ndarray, np.ndarray]:
    """
    Demo event function - generate and load images.
    """
    logger.info("Generation started...")
    t = time.time()
    IMAGE_OBJ.load_from_generator(cast(RxImageGenerator, GENERATOR_OBJ))
    logger.info(f"...Generation finished. Spent {time.time()-t:.2f} seconds.")
    return IMAGE_OBJ.image, IMAGE_OBJ.ref_image


def demo_detect() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Demo event function - detect trigger button.
    """
    logger.info("Detection started...")
    t = time.time()
    ROI, BB = IMAGE_OBJ.ROIs, IMAGE_OBJ.draw_bounding_boxes()
    logger.info(f"...Detection finished. Spent {time.time()-t:.2f} seconds.")
    return ROI, BB


def demo_segment() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Demo event function - segment trigger button.
    """
    logger.info("Segmentation started...")
    t = time.time()
    mROI, mFull = IMAGE_OBJ.masked_ROIs, IMAGE_OBJ.background_segment
    logger.info(f"...Segmentation finished. Spent {time.time()-t:.2f} seconds.")
    return mROI, mFull * 255


def demo_verify() -> list[tuple[np.ndarray, str]]:
    """
    Demo event function - verify trigger button.
    """
    logger.info("Verification started...")
    t = time.time()
    verify_scores = [(ROI, f"{score:.3f}") for ROI, score in zip(IMAGE_OBJ.ROIs, IMAGE_OBJ.similarity_scores)]
    logger.info(f"...Verification finished. Spent {time.time()-t:.2f} seconds.")
    return verify_scores


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
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# RxImage")
        with gr.Tab("Image Loading"):
            with gr.Row():
                with gr.Column():
                    image_source = gr.Radio(
                        choices=["Camera", "Upload"], value="Upload", label="Choose Image Source"
                    )
                    image_input = gr.Image(label="Image to verify", shape=(200, 200))
                image_ref = gr.Image(label="Reference image", shape=(200, 200))
            with gr.Row():
                image_load_button = gr.Button("Load")
                image_gen_button = gr.Button("Generate (for testing purposes)")
        with gr.Tab("Detection"):
            with gr.Row():
                gallery_ROI = gr.Gallery(label="ROIs", columns=5, object_fit="scale-down")
                image_BB = gr.Image(label="Bounding boxes", shape=(200, 200))
            image_detect_button = gr.Button("Detect")
        with gr.Tab("Segmentation"):
            with gr.Row():
                gallery_mROI = gr.Gallery(label="Masked ROIs", columns=5, object_fit="scale-down")
                image_mFull = gr.Image(label="Bachground segmentations", shape=(200, 200))
            image_segment_button = gr.Button("Segment")
        with gr.Tab("Verification"):
            gallery_scores = gr.Gallery(label="Verification Scores", columns=5, object_fit="scale-down")
            image_verify_button = gr.Button("Verify")
        with gr.Tab("Counting Helper"):
            with gr.Row():
                streaming_source = gr.Image(label="Input camera stream", source="webcam", visible=False)
                CountHelpBox = gr.Textbox(show_label=False)
                streaming_destination = gr.Image(label="Output result")

        with gr.Accordion("Logs"):
            logBox = gr.Textbox(show_label=False)

        image_source.change(demo_choose_source, image_source, image_input)
        image_load_button.click(demo_load_image, inputs=[image_input, image_ref], outputs=None)
        image_gen_button.click(demo_gen_image, inputs=None, outputs=[image_input, image_ref])
        image_detect_button.click(demo_detect, None, [gallery_ROI, image_BB])
        image_segment_button.click(demo_segment, None, [gallery_mROI, image_mFull])
        image_verify_button.click(demo_verify, None, gallery_scores)
        demo.queue()
        demo.load(demo_init)
        streaming_source.stream(
            demo_video_counter,
            streaming_source,
            [streaming_destination, CountHelpBox],
            show_progress="hidden",
        )
        demo.load(demo_read_logs, None, logBox, every=1)
    return demo


if __name__ == "__main__":
    create_demo().launch(share=True)
