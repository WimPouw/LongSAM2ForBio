#!/usr/bin/env python3
"""
Standalone inference script for trained Detectron2 models.
Usage: python inference.py --model path/to/model.pth --input path/to/video.mp4 --output path/to/output.mp4
"""

import argparse
import cv2
import torch
import json
import os
from pathlib import Path
import logging

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

setup_logger()

class ObjectTracker:
    """
    Simple object tracking using Detectron2 predictions.
    """
    
    def __init__(self, model_path: str, config_file: str, num_classes: int, 
                 confidence_threshold: float = 0.7, device: str = "cuda"):
        """
        Initialize the tracker.
        
        Args:
            model_path: Path to trained model weights
            config_file: Detectron2 config file name
            num_classes: Number of object classes
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.DEVICE = device
        
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0] if self.cfg.DATASETS.TEST else "")
        
        logging.info(f"Model loaded successfully from {model_path}")
        logging.info(f"Running on device: {device}")
    
    def detect_objects(self, image):
        """
        Detect objects in a single image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Detectron2 outputs dictionary
        """
        return self.predictor(image)
    
    def visualize_predictions(self, image, outputs):
        """
        Create visualization of predictions on image.
        
        Args:
            image: Input image (numpy array)
            outputs: Detectron2 outputs
            
        Returns:
            Visualization image
        """
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]
    
    def process_video(self, input_path: str, output_path: str, 
                     save_detections: bool = True, show_progress: bool = True):
        """
        Process entire video and save results.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            save_detections: Whether to save detection results as JSON
            show_progress: Whether to show processing progress
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track all detections
        all_detections = []
        frame_idx = 0
        
        logging.info(f"Processing video: {input_path}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            outputs = self.detect_objects(frame)
            
            # Visualize predictions
            vis_frame = self.visualize_predictions(frame, outputs)
            out.write(vis_frame)
            
            # Save detection data
            if save_detections:
                frame_data = self.extract_detection_data(outputs, frame_idx)
                all_detections.append(frame_data)
            
            frame_idx += 1
            
            # Show progress
            if show_progress and frame_idx % 100 == 0:
                logging.info(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save detections as JSON
        if save_detections:
            detections_path = output_path.replace('.mp4', '_detections.json')
            with open(detections_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            logging.info(f"Detections saved to: {detections_path}")
        
        logging.info(f"Video processing complete: {output_path}")
        return all_detections if save_detections else None
    
    def extract_detection_data(self, outputs, frame_idx):
        """
        Extract detection data from Detectron2 outputs.
        
        Args:
            outputs: Detectron2 prediction outputs
            frame_idx: Current frame index
            
        Returns:
            Dictionary with detection data
        """
        instances = outputs["instances"].to("cpu")
        
        frame_data = {
            "frame": frame_idx,
            "detections": []
        }
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for i in range(len(instances)):
                detection = {
                    "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                    "score": float(scores[i]),
                    "class_id": int(classes[i]),
                    "area": float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                }
                frame_data["detections"].append(detection)
        
        return frame_data
    
    def process_image(self, input_path: str, output_path: str):
        """
        Process single image and save result.
        
        Args:
            input_path: Path to input image
            output_path: Path to output image
        """
        image = cv2.imread(input_path)
        outputs = self.detect_objects(image)
        vis_image = self.visualize_predictions(image, outputs)
        cv2.imwrite(output_path, vis_image)
        
        logging.info(f"Image processed: {input_path} -> {output_path}")
        return self.extract_detection_data(outputs, 0)


def main():
    parser = argparse.ArgumentParser(description="Object detection inference using trained Detectron2 model")
    parser.add_argument("--model", required=True, help="Path to trained model weights (.pth file)")
    parser.add_argument("--input", required=True, help="Path to input video or image")
    parser.add_argument("--output", required=True, help="Path to output video or image")
    parser.add_argument("--config", default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", 
                       help="Detectron2 config file")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of object classes")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--save-detections", action="store_true", help="Save detection results as JSON")
    parser.add_argument("--no-progress", action="store_true", help="Don't show progress updates")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = ObjectTracker(
        model_path=args.model,
        config_file=args.config,
        num_classes=args.num_classes,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Determine if input is video or image
    input_ext = Path(args.input).suffix.lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if input_ext in video_extensions:
        # Process video
        detections = tracker.process_video(
            input_path=args.input,
            output_path=args.output,
            save_detections=args.save_detections,
            show_progress=not args.no_progress
        )
        
        if detections:
            total_detections = sum(len(frame["detections"]) for frame in detections)
            logging.info(f"Total detections across all frames: {total_detections}")
    
    elif input_ext in image_extensions:
        # Process image
        detection = tracker.process_image(args.input, args.output)
        logging.info(f"Detections in image: {len(detection['detections'])}")
        
        if args.save_detections:
            detections_path = args.output.replace(input_ext, '_detections.json')
            with open(detections_path, 'w') as f:
                json.dump([detection], f, indent=2)
            logging.info(f"Detections saved to: {detections_path}")
    
    else:
        logging.error(f"Unsupported file format: {input_ext}")
        return


if __name__ == "__main__":
    main()