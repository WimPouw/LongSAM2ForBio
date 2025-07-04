#!/usr/bin/env python3
"""
Complete Video Analysis Script with SAM2
Combines video processing, segmentation, and analysis into a single script
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import shutil
from pathlib import Path
import json
from datetime import datetime
import gc
from tqdm import tqdm
import pandas as pd

# Configure environment
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def setup_device():
    """Setup computation device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\nSupport for MPS devices is preliminary. SAM 2 might give numerically different outputs on MPS.")
    
    return device

def get_video_fps(video_path):
    """Get video FPS using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def video_to_frames(input_video, output_dir, quality=2):
    """Convert video to frames using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    fps, total_frames = get_video_fps(input_video)
    print(f"Video: {Path(input_video).name}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y to overwrite existing files
        '-i', input_video,
        '-q:v', str(quality),
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            print(f"Successfully created {num_frames} frames")
            return fps, num_frames
        else:
            print(f"Error: {result.stderr}")
            return -1, -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1, -1

class VideoChunkProcessor:
    def __init__(self, predictor, video_dir, chunk_size=500, **kwargs):
        self.predictor = predictor
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Video directory {self.video_dir} does not exist!")
        
        self.frame_names = sorted(
            [p for p in os.listdir(self.video_dir) 
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        if not self.frame_names:
            raise ValueError("No frames found in the specified directory!")
        
        self.chunks = self._calculate_chunks()
        print(f"Created {len(self.chunks)} chunks of {chunk_size} frames each")

    def _calculate_chunks(self):
        """Calculate chunks without overlap"""
        chunks = []
        frame_count = len(self.frame_names)
        
        for start in range(0, frame_count, self.chunk_size):
            end = min(start + self.chunk_size, frame_count)
            chunks.append({
                'start': start,
                'end': end,
                'frame_names': self.frame_names[start:end]
            })
        return chunks

    def _get_last_valid_masks(self, results, frame_idx, reverse_search_limit=30):
        """Get masks from the last valid frame"""
        for i in range(frame_idx, max(frame_idx - reverse_search_limit, -1), -1):
            if i in results and results[i]:
                return results[i]
        return None

    def _compute_box_from_mask(self, mask):
        """Compute bounding box from mask"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask = mask.astype(bool)
        
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
            
        padding = 10
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        x0 = max(0, x0 - padding)
        x1 = x1 + padding
        y0 = max(0, y0 - padding)
        y1 = y1 + padding
        
        return [int(x0), int(y0), int(x1), int(y1)]

    def _compute_metrics(self, mask, box, frame):
        """Compute metrics from mask, bounding box, and frame"""
        if box is None or mask is None or frame is None:
            return None
        
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask = mask.astype(bool)
        
        x1, y1, x2, y2 = box
        box_centroid_x = (x1 + x2) / 2
        box_centroid_y = (y1 + y2) / 2
        
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
            
        seg_centroid_y = np.mean(y_coords)
        seg_centroid_x = np.mean(x_coords)
        surface_area = np.sum(mask)
        
        try:
            mask_3d = np.stack([mask] * 3, axis=-1)
            masked_frame = np.where(mask_3d, frame, 0)
            valid_pixels = masked_frame[mask]
            
            if len(valid_pixels) > 0:
                mean_color = np.mean(valid_pixels, axis=0)
                std_color = np.std(valid_pixels, axis=0)
            else:
                return None
                
            return {
                'box_x1': x1, 'box_y1': y1, 'box_x2': x2, 'box_y2': y2,
                'box_centroid_x': box_centroid_x, 'box_centroid_y': box_centroid_y,
                'seg_centroid_y': float(seg_centroid_y), 'seg_centroid_x': float(seg_centroid_x),
                'surface_area': int(surface_area),
                'mean_color_b': float(mean_color[0]), 'mean_color_g': float(mean_color[1]), 'mean_color_r': float(mean_color[2]),
                'std_color_b': float(std_color[0]), 'std_color_g': float(std_color[1]), 'std_color_r': float(std_color[2]),
                'color_intensity': float(np.mean(mean_color))
            }
        except Exception as e:
            print(f"Error in color analysis: {str(e)}")
            return None

    def _get_mask_boundary_points(self, mask, num_boundary_points=4, num_interior_points=6, num_negative_points=24, debug_viz=False):
        """Extract points from a binary mask for propagation"""
        if not mask.any():
            return None, None
            
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        points = []
        labels = []
        
        # Boundary points
        if num_boundary_points > 0:
            leftmost = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
            topmost = tuple(largest_contour[largest_contour[:,:,1].argmin()][0])
            bottommost = tuple(largest_contour[largest_contour[:,:,1].argmax()][0])
            
            extremes = [leftmost, rightmost, topmost, bottommost]
            points.extend(extremes[:num_boundary_points])
            labels.extend([1] * num_boundary_points)
        
        # Interior points
        if num_interior_points > 0:
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
        
                if mask[cy, cx]:
                    points.append([cx, cy])
                    labels.append(1)
        
                max_radius = min(w, h) // 4
                angles = np.linspace(0, 2 * np.pi, num_interior_points)
                radii = np.linspace(max_radius * 0.2, max_radius, 3)
        
                for radius in radii:
                    for angle in angles:
                        x_pt = cx + int(radius * np.cos(angle))
                        y_pt = cy + int(radius * np.sin(angle))
                        
                        if (0 <= x_pt < mask.shape[1] and 0 <= y_pt < mask.shape[0] and mask[y_pt, x_pt]):
                            points.append([x_pt, y_pt])
                            labels.append(1)
        
        # Negative points
        if num_negative_points > 0:
            expansion = max(w, h) // 10
            kernel = np.ones((expansion, expansion), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            outer_region = dilated & (~mask)
            
            ys, xs = np.where(outer_region > 0)
            if len(xs) > 0:
                indices = np.linspace(0, len(xs) - 1, num_negative_points, dtype=int)
                negative_points = np.column_stack([xs[indices], ys[indices]])
                
                for i in range(len(negative_points)):
                    offset = np.random.randint(-expansion//2, expansion//2, size=2)
                    pt = negative_points[i] + offset
                    pt[0] = np.clip(pt[0], 0, mask.shape[1]-1)
                    pt[1] = np.clip(pt[1], 0, mask.shape[0]-1)
                    if not mask[pt[1], pt[0]]:
                        points.append(pt)
                        labels.append(0)
        
        if not points:
            return None, None
            
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    def process_video(self, points_dict, labels_dict, debug=True):
        """Process video in chunks"""
        results = {}
        
        def cleanup_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        def process_chunk(chunk_idx, prev_masks=None):
            chunk = self.chunks[chunk_idx]
            chunk_results = {}
            temp_dir = os.path.join(self.video_dir, f"chunk_{chunk_idx}")
            
            try:
                os.makedirs(temp_dir, exist_ok=True)
                for frame_name in chunk['frame_names']:
                    src = os.path.join(self.video_dir, frame_name)
                    dst = os.path.join(temp_dir, frame_name)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                
                chunk_state = self.predictor.init_state(video_path=temp_dir)
                
                for obj_id in points_dict:
                    try:
                        self.predictor.reset_state(chunk_state)
                        
                        if chunk_idx == 0:
                            points = np.array(points_dict[obj_id], dtype=np.float32)
                            labels = np.array(labels_dict[obj_id], dtype=np.int32)
                        else:
                            if prev_masks is None or obj_id not in prev_masks:
                                continue
                                
                            mask = prev_masks[obj_id]
                            if len(mask.shape) == 3:
                                mask = mask[0]
                            
                            points, labels = self._get_mask_boundary_points(mask, debug_viz=debug)
                            if points is None:
                                continue
                        
                        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=chunk_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            points=points,
                            labels=labels
                        )
                        
                        for i, prop_obj_id in enumerate(obj_ids):
                            mask = (mask_logits[i] > 0.0).cpu().numpy()
                            if len(mask.shape) == 3:
                                mask = mask[0]
                            frame_idx = chunk['start']
                            if frame_idx not in chunk_results:
                                chunk_results[frame_idx] = {}
                            chunk_results[frame_idx][prop_obj_id] = mask.copy()
                        
                        del mask_logits
                        cleanup_memory()
                        
                        for frame_idx, prop_obj_ids, prop_mask_logits in self.predictor.propagate_in_video(chunk_state):
                            global_frame_idx = chunk['start'] + frame_idx
                            
                            for i, prop_obj_id in enumerate(prop_obj_ids):
                                mask = (prop_mask_logits[i] > 0.0).cpu().numpy()
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                if global_frame_idx not in chunk_results:
                                    chunk_results[global_frame_idx] = {}
                                chunk_results[global_frame_idx][prop_obj_id] = mask.copy()
                            
                            del prop_mask_logits
                            cleanup_memory()
                    
                    finally:
                        cleanup_memory()
                
                return chunk_results
                
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                cleanup_memory()
        
        try:
            cleanup_memory()
            
            print("\nProcessing first chunk...")
            first_chunk_results = process_chunk(0)
            results.update(first_chunk_results)
            
            for chunk_idx in range(1, len(self.chunks)):
                print(f"\nProcessing chunk {chunk_idx}...")
                
                prev_chunk = self.chunks[chunk_idx - 1]
                prev_last_frame = prev_chunk['end'] - 1
                prev_masks = self._get_last_valid_masks(results, prev_last_frame)
                
                if prev_masks is None:
                    continue
                
                chunk_results = process_chunk(chunk_idx, prev_masks)
                if chunk_results:
                    results.update(chunk_results)
                
                cleanup_memory()
            
            return results
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            return None
        finally:
            cleanup_memory()

    def save_results_video(self, results, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results as video"""
        if not results:
            print("No results to save!")
            return
    
        fps = float(fps)
        alpha = max(0.0, min(1.0, alpha))
        
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]
    
        cmap = plt.get_cmap("tab10")
        
        if show_original:
            out_width = width * 2
        else:
            out_width = width
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
    
        print("\nSaving video...")
        for frame_idx in tqdm(range(len(self.frame_names))):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            if frame is None:
                continue
                
            overlay = frame.copy()
            
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    
                    if mask.shape != (height, width) and mask.shape[0] > 0 and mask.shape[1] > 0:
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), 
                                            interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue
                    
                    if mask.shape == (height, width):
                        color = np.array(cmap(obj_id % 10)[:3]) * 255
                        
                        if alpha == 1.0:
                            for c in range(3):
                                overlay[:, :, c][mask] = color[c]
                        else:
                            color_mask = np.zeros_like(overlay)
                            for c in range(3):
                                color_mask[:, :, c][mask] = color[c]
                            
                            blend_mask = np.zeros_like(overlay)
                            cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0, blend_mask)
                            overlay[mask] = blend_mask[mask]
            
            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay
                
            out.write(output_frame)
    
        out.release()
        print(f"Video saved to: {output_path}")

    def _save_time_series(self, csv_path):
        """Save time series metrics"""
        metrics_data = []
        
        for frame_idx in sorted(self.results.keys()):
            frame_name = self.frame_names[frame_idx]
            try:
                frame = cv2.imread(os.path.join(self.video_dir, frame_name))
                if frame is None:
                    continue
                    
                for obj_id, mask in self.results[frame_idx].items():
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    
                    box = self._compute_box_from_mask(mask)
                    if box is None:
                        continue
                        
                    metrics = self._compute_metrics(mask, box, frame)
                    if metrics is None:
                        continue
                    
                    metrics.update({
                        'frame': frame_idx,
                        'frame_name': frame_name,
                        'object_id': obj_id
                    })
                    
                    metrics_data.append(metrics)
                    
            except Exception as e:
                continue
        
        if not metrics_data:
            print("No valid metrics data collected")
            return
            
        df = pd.DataFrame(metrics_data)
        df['delta_centroid_x'] = df.groupby('object_id')['seg_centroid_x'].diff()
        df['delta_centroid_y'] = df.groupby('object_id')['seg_centroid_y'].diff()
        df['delta_area'] = df.groupby('object_id')['surface_area'].diff()
        
        df.to_csv(csv_path, index=False)
        print(f"Saved time series metrics to: {csv_path}")

    def _save_coco_annotations(self, json_path):
        """Save annotations in COCO format"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "SAM2 segmentation results",
                "date_created": current_time
            },
            "images": [],
            "annotations": [],
            "licenses": [{"id": 0, "name": "Unknown License", "url": ""}],
            "categories": [{"supercategory": "object", "id": 1, "name": "object"}]
        }
        
        # Add images
        unique_ids = {}
        for idx, frame_name in enumerate(self.frame_names, 1):
            img = cv2.imread(os.path.join(self.video_dir, frame_name))
            height, width = img.shape[:2]
            coco_data["images"].append({
                "id": idx,
                "width": width,
                "height": height,
                "file_name": frame_name,
                "license": 0,
                "date_captured": ""
            })
            unique_ids[frame_name] = idx
        
        # Add annotations
        annotation_id = 1
        for frame_idx in self.results:
            current_frame = self.frame_names[frame_idx]
            for obj_id, mask in self.results[frame_idx].items():
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                mask_bool = mask.astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    flattened = []
                    for point in approx:
                        flattened.extend([int(point[0][0]), int(point[0][1])])
                    
                    box = self._compute_box_from_mask(mask)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        area = int(cv2.contourArea(contour))
                        
                        coco_data["annotations"].append({
                            "segmentation": [flattened],
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "id": annotation_id,
                            "image_id": unique_ids[current_frame],
                            "category_id": 1
                        })
                        annotation_id += 1
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)
        print(f"Saved COCO annotations to: {json_path}")

    def save_results(self, output_path, fps=30, show_original=True, alpha=0.5):
        """Save all results"""
        self.save_results_video(self.results, output_path, fps, show_original, alpha)
        self._save_coco_annotations(os.path.join(os.path.dirname(output_path), "segmentation_coco.json"))
        self._save_time_series(os.path.join(os.path.dirname(output_path), "time_series_metrics.csv"))

def select_points_opencv(frame, processor=None):
    """Interactive point selection tool"""
    points_dict = {}
    labels_dict = {}
    current_obj_id = 1
    
    temp_dir = "temp_select"
    if processor is not None:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            frame_path = os.path.join(temp_dir, "0.jpg")
            cv2.imwrite(frame_path, frame)
            
            chunk_state = processor.predictor.init_state(video_path=temp_dir)
            if chunk_state is None:
                raise ValueError("Failed to initialize chunk state")
            
        except Exception as e:
            print(f"Error initializing processor: {str(e)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, None
    
    def draw_point(img, point, obj_id, label):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
        cv2.putText(img, str(obj_id), 
                   (int(point[0] + 5), int(point[1] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def redraw_all_points():
        display = frame.copy()
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        return display
    
    def test_mask():
        try:
            if not points_dict or not points_dict.get(current_obj_id):
                print("No points selected for current object")
                return
            
            if processor is None or chunk_state is None:
                print("Processor not properly initialized")
                return
            
            points = np.array(points_dict[current_obj_id], dtype=np.float32)
            labels = np.array(labels_dict[current_obj_id], dtype=np.int32)
            
            processor.predictor.reset_state(chunk_state)
            _, obj_ids, mask_logits = processor.predictor.add_new_points_or_box(
                inference_state=chunk_state,
                frame_idx=0,
                obj_id=current_obj_id,
                points=points,
                labels=labels
            )
            
            if len(mask_logits) > 0:
                mask = (mask_logits[0] > 0.0).cpu().numpy()
                if len(mask.shape) == 3:
                    mask = mask[0]
                
                height, width = frame.shape[:2]
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height),
                                    interpolation=cv2.INTER_LINEAR) > 0.5
                
                preview = frame.copy()
                color = np.array(plt.get_cmap("tab10")(current_obj_id % 10)[:3]) * 255
                
                color_overlay = np.zeros_like(preview)
                for c in range(3):
                    color_overlay[:, :, c][mask] = color[c]
                
                preview = cv2.addWeighted(preview, 0.7, color_overlay, 0.3, 0)
                
                for obj_id in points_dict:
                    for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                        draw_point(preview, pt, obj_id, label)
                
                cv2.namedWindow('Mask Preview', cv2.WINDOW_NORMAL)
                cv2.imshow('Mask Preview', preview)
                cv2.waitKey(1)
                
        except Exception as e:
            print(f"Error in test_mask: {str(e)}")
        
    def click_handler(event, x, y, flags, param):
        nonlocal img_display
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if current_obj_id not in points_dict:
                points_dict[current_obj_id] = []
                labels_dict[current_obj_id] = []
            
            points_dict[current_obj_id].append([x, y])
            label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            labels_dict[current_obj_id].append(label)
            
            draw_point(img_display, [x, y], current_obj_id, label)
            print(f"Added {'positive' if label == 1 else 'negative'} point for object {current_obj_id}")
    
    img_display = frame.copy()
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', click_handler)
    
    print("\nControls:")
    print("- Left click: add positive point (green)")
    print("- Right click: add negative point (red)")
    print("- Press 'r' to reset current object")
    print("- Press 'n' for next object")
    print("- Press 'p' for previous object")
    print("- Press 't' to test mask")
    print("- Press Enter to finish")
    print("- Press 'q' to quit")
    
    while True:
        cv2.imshow('Select Points', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            points_dict[current_obj_id] = []
            labels_dict[current_obj_id] = []
            img_display = redraw_all_points()
            print(f"Reset points for object {current_obj_id}")
        
        elif key == ord('n'):
            current_obj_id += 1
            print(f"Now selecting object {current_obj_id}")
        
        elif key == ord('p'):
            if current_obj_id > 1:
                current_obj_id -= 1
                print(f"Now selecting object {current_obj_id}")
        
        elif key == ord('t'):
            test_mask()
        
        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return points_dict, labels_dict if points_dict else (None, None)
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, None
    
    return points_dict, labels_dict

class VideoAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Analysis with SAM2")
        self.root.geometry("400x300")
        
        # Initialize SAM2
        self.device = setup_device()
        self.predictor = None
        self.init_sam2()
        
        self.setup_gui()
        
    def init_sam2(self):
        """Initialize SAM2 predictor"""
        try:
            # You need to update these paths to your SAM2 installation
            sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if not os.path.exists(sam2_checkpoint):
                messagebox.showerror("Error", 
                    f"SAM2 checkpoint not found at: {sam2_checkpoint}\n"
                    "Please update the path in the script.")
                return
            
            from sam2.build_sam import build_sam2_video_predictor
            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            print("SAM2 predictor initialized successfully")
            
        except ImportError:
            messagebox.showerror("Error", 
                "SAM2 not found. Please install SAM2 first:\n"
                "git clone https://github.com/facebookresearch/segment-anything-2.git\n"
                "cd segment-anything-2\n"
                "pip install -e .")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize SAM2: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI"""
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="Video Analysis with SAM2", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Folder selection
        folder_frame = tk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(folder_frame, text="Select folder containing videos:", 
                font=("Arial", 10)).pack(anchor=tk.W)
        
        self.folder_var = tk.StringVar()
        folder_entry = tk.Entry(folder_frame, textvariable=self.folder_var, 
                               width=40, state='readonly')
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Button(folder_frame, text="Browse", 
                 command=self.select_folder).pack(side=tk.RIGHT)
        
        # Video list
        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(list_frame, text="Videos found:", 
                font=("Arial", 10)).pack(anchor=tk.W)
        
        listbox_frame = tk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.video_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.video_listbox.yview)
        
        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=20)
        
        tk.Button(process_frame, text="Process Selected Video", 
                 command=self.process_video, bg="#4CAF50", fg="white",
                 font=("Arial", 12, "bold"), pady=10).pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 9))
        status_label.pack(pady=(10, 0))
    
    def select_folder(self):
        """Select folder containing videos"""
        folder = filedialog.askdirectory(title="Select folder containing videos")
        if folder:
            self.folder_var.set(folder)
            self.scan_videos(folder)
    
    def scan_videos(self, folder):
        """Scan for video files in folder"""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']
        videos = []
        
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(file)
        
        videos.sort()
        
        self.video_listbox.delete(0, tk.END)
        for video in videos:
            self.video_listbox.insert(tk.END, video)
        
        if videos:
            self.video_listbox.select_set(0)  # Select first video
            self.status_var.set(f"Found {len(videos)} video(s)")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number(self, total_frames):
        """Get frame number for mask selection"""
        frame_num = simpledialog.askinteger(
            "Frame Selection",
            f"Enter frame number for mask selection (0-{total_frames-1}):",
            minvalue=0,
            maxvalue=total_frames-1,
            initialvalue=min(250, total_frames//2)
        )
        return frame_num
    
    def process_video(self):
        """Process the selected video"""
        if self.predictor is None:
            messagebox.showerror("Error", "SAM2 predictor not initialized")
            return
        
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a video to process")
            return
        
        folder = self.folder_var.get()
        if not folder:
            messagebox.showwarning("Warning", "Please select a folder first")
            return
        
        video_name = self.video_listbox.get(selection[0])
        video_path = os.path.join(folder, video_name)
        
        try:
            self.status_var.set("Processing video...")
            self.root.update()
            
            # Create frames directory
            video_stem = Path(video_name).stem
            frames_dir = os.path.join(folder, f"{video_stem}_frames")
            
            # Extract frames
            self.status_var.set("Extracting frames...")
            self.root.update()
            
            fps, num_frames = video_to_frames(video_path, frames_dir)
            if fps == -1:
                messagebox.showerror("Error", "Failed to extract frames from video")
                return
            
            # Get frame number for mask selection
            frame_num = self.get_frame_number(num_frames)
            if frame_num is None:
                return
            
            # Initialize processor
            self.status_var.set("Initializing processor...")
            self.root.update()
            
            processor = VideoChunkProcessor(self.predictor, frames_dir, chunk_size=500)
            
            # Load the selected frame for mask selection
            frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
            if not os.path.exists(frame_path):
                messagebox.showerror("Error", f"Frame {frame_num} not found")
                return
            
            frame = cv2.imread(frame_path)
            
            # Point selection
            self.status_var.set("Select points on the frame...")
            self.root.update()
            
            messagebox.showinfo("Point Selection", 
                "The frame will open in a new window.\n"
                "Follow the instructions in the console for point selection.")
            
            points_dict, labels_dict = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Process video
            self.status_var.set("Processing video with SAM2...")
            self.root.update()
            
            results = processor.process_video(points_dict, labels_dict)
            
            if results:
                processor.results = results
                
                # Save results
                self.status_var.set("Saving results...")
                self.root.update()
                
                output_path = os.path.join(frames_dir, "output_masked.mp4")
                processor.save_results(
                    output_path=output_path,
                    fps=fps,
                    show_original=True,
                    alpha=0.5
                )
                
                self.status_var.set("Processing completed successfully!")
                messagebox.showinfo("Success", 
                    f"Processing completed!\n"
                    f"Results saved in: {frames_dir}\n"
                    f"- Masked video: output_masked.mp4\n"
                    f"- COCO annotations: segmentation_coco.json\n"
                    f"- Time series data: time_series_metrics.csv")
            else:
                messagebox.showerror("Error", "Video processing failed")
                self.status_var.set("Processing failed")
        
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting Video Analysis Application...")
    print("Make sure you have:")
    print("1. SAM2 installed and checkpoints downloaded")
    print("2. FFmpeg installed for video processing")
    print("3. Required Python packages: opencv-python, torch, matplotlib, pandas, tqdm")
    print("\nStarting GUI...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()