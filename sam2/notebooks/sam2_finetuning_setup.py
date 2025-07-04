#!/usr/bin/env python3
"""
Complete SAM2 Video Analysis Script - Fixed Version
Addresses all major issues: frame preview, backward propagation, analysis video, chunk size UI, finetuning setup
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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

def cleanup_memory():
    """Clean up GPU/CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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

def show_frame_preview(frames_dir, frame_idx, total_frames):
    """Show a preview of the selected frame"""
    frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
    if not os.path.exists(frame_path):
        messagebox.showerror("Error", f"Frame {frame_idx} not found")
        return False
    
    frame = cv2.imread(frame_path)
    if frame is None:
        messagebox.showerror("Error", f"Could not load frame {frame_idx}")
        return False
    
    # Resize frame for preview if too large
    height, width = frame.shape[:2]
    max_size = 800
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Add frame info text
    info_text = f"Frame {frame_idx}/{total_frames-1} - Preview"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press any key to continue...", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.namedWindow('Frame Preview', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame Preview', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('Frame Preview')
    return True

class VideoChunkProcessor:
    def __init__(self, predictor, video_dir, chunk_size=500, overlap_frames=20, 
                 interactive_correction=True, seed_frame_idx=0):
        self.predictor = predictor
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.interactive_correction = interactive_correction
        self.seed_frame_idx = seed_frame_idx
        
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Video directory {self.video_dir} does not exist!")
        
        self.frame_names = sorted(
            [p for p in os.listdir(self.video_dir) 
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        if not self.frame_names:
            raise ValueError("No frames found in the specified directory!")
        
        # Validate seed frame
        if self.seed_frame_idx >= len(self.frame_names):
            self.seed_frame_idx = len(self.frame_names) // 2
            print(f"⚠️ Seed frame index too high, using middle frame: {self.seed_frame_idx}")
        
        print(f"Created processor with chunk size {chunk_size}, seed frame at {self.seed_frame_idx}")
        if interactive_correction:
            print("🎯 Interactive correction mode enabled")

    def _create_temp_video_dir(self, frames, temp_dir_name):
        """Create temporary directory with numbered frames for SAM2"""
        temp_dir = os.path.join(self.video_dir, temp_dir_name)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        for i, frame_name in enumerate(frames):
            src = os.path.join(self.video_dir, frame_name)
            dst = os.path.join(temp_dir, f"{i:05d}.jpg")
            shutil.copy2(src, dst)
        
        return temp_dir

    def _generate_robust_points_from_mask(self, mask, num_positive=8, num_negative=16):
        """Generate robust points from mask for propagation"""
        if not mask.any():
            return None, None
            
        points = []
        labels = []
        
        if len(mask.shape) == 3:
            mask = mask[0]
        mask = mask.astype(bool)
        
        # Get contour points
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_length = cv2.arcLength(largest_contour, True)
            if contour_length > 0:
                spacing = max(1, int(contour_length / num_positive))
                for i in range(0, len(largest_contour), spacing):
                    if len(points) >= num_positive:
                        break
                    point = largest_contour[i][0]
                    points.append([point[0], point[1]])
                    labels.append(1)
        
        # Add center point
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            if mask[cy, cx]:
                points.append([cx, cy])
                labels.append(1)
        
        # Add negative points around the mask
        kernel_size = max(10, int(np.sqrt(np.sum(mask)) * 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        negative_region = expanded & (~mask)
        
        neg_y, neg_x = np.where(negative_region)
        if len(neg_x) > 0:
            neg_indices = np.random.choice(len(neg_x), min(num_negative, len(neg_x)), replace=False)
            for idx in neg_indices:
                points.append([neg_x[idx], neg_y[idx]])
                labels.append(0)
        
        if not points:
            return None, None
            
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    def process_video(self, points_dict, labels_dict, debug=True):
        """Process video with improved bidirectional propagation"""
        results = {}
        
        try:
            cleanup_memory()
            
            # Step 1: Process seed frame
            print(f"\n🎯 Step 1: Processing seed frame {self.seed_frame_idx}")
            seed_results = self._process_seed_frame(points_dict, labels_dict, debug)
            
            if not seed_results:
                print("❌ Failed to process seed frame")
                return None
            
            results.update(seed_results)
            print(f"✅ Seed processing complete")
            
            # Step 2: Forward propagation (seed → end)
            print(f"\n➡️ Step 2: Forward propagation")
            forward_results = self._process_forward_propagation(seed_results, debug)
            results.update(forward_results)
            print(f"✅ Forward propagation complete: {len(forward_results)} frames")
            
            # Step 3: Backward propagation (seed → start) - FIXED
            print(f"\n⬅️ Step 3: Backward propagation")
            backward_results = self._process_backward_propagation(seed_results, debug)
            results.update(backward_results)
            print(f"✅ Backward propagation complete: {len(backward_results)} frames")
            
            # Step 4: Fill gaps
            self._fill_result_gaps(results, debug)
            print(f"\n🎉 Processing complete! Total frames: {len(results)}/{len(self.frame_names)}")
            
            return results
            
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cleanup_memory()

    def _process_seed_frame(self, points_dict, labels_dict, debug=True):
        """Process the seed frame with user annotations"""
        seed_results = {}
        temp_dir = "temp_seed"
        
        try:
            # Create temp directory with seed frame
            seed_frame_name = self.frame_names[self.seed_frame_idx]
            temp_dir_path = self._create_temp_video_dir([seed_frame_name], temp_dir)
            
            chunk_state = self.predictor.init_state(video_path=temp_dir_path)
            
            # Process each object
            for obj_id in points_dict:
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    points = np.array(points_dict[obj_id], dtype=np.float32)
                    labels = np.array(labels_dict[obj_id], dtype=np.int32)
                    
                    if debug:
                        print(f"  Object {obj_id}: +{sum(labels == 1)} -{sum(labels == 0)} points")
                    
                    # Add prompts to frame 0 (the seed frame)
                    _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=chunk_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
                    
                    # Store result
                    for i, prop_obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        
                        if self.seed_frame_idx not in seed_results:
                            seed_results[self.seed_frame_idx] = {}
                        seed_results[self.seed_frame_idx][prop_obj_id] = mask.copy()
                    
                    del mask_logits
                    cleanup_memory()
                
                except Exception as e:
                    print(f"  Error processing object {obj_id}: {e}")
                    continue
            
            return seed_results
            
        except Exception as e:
            print(f"Error processing seed frame: {e}")
            return {}
        finally:
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)
            cleanup_memory()

    def _process_forward_propagation(self, seed_results, debug=True):
        """Process frames forward from seed to end"""
        forward_results = {}
        
        # Get frames after seed
        forward_frames = self.frame_names[self.seed_frame_idx + 1:]
        if not forward_frames:
            return forward_results
        
        # Process in chunks
        for chunk_start in range(0, len(forward_frames), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(forward_frames))
            chunk_frames = forward_frames[chunk_start:chunk_end]
            
            # Add seed frame at the beginning for reference
            chunk_with_seed = [self.frame_names[self.seed_frame_idx]] + chunk_frames
            
            if debug:
                print(f"  Forward chunk: frames {self.seed_frame_idx + 1 + chunk_start} to {self.seed_frame_idx + chunk_end}")
            
            chunk_results = self._process_chunk(chunk_with_seed, seed_results, is_forward=True, debug=debug)
            
            # Remove seed frame from results (already have it)
            chunk_results.pop(self.seed_frame_idx, None)
            forward_results.update(chunk_results)
        
        return forward_results

    def _process_backward_propagation(self, seed_results, debug=True):
        """Process frames backward from seed to start - FIXED VERSION"""
        backward_results = {}
        
        # Get frames before seed
        backward_frames = self.frame_names[:self.seed_frame_idx]
        if not backward_frames:
            return backward_results
        
        # IMPORTANT: Process backward frames in REVERSE ORDER
        # This is the key fix - we reverse the frames so propagation goes backward in time
        backward_frames_reversed = backward_frames[::-1]
        
        # Process in chunks
        for chunk_start in range(0, len(backward_frames_reversed), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(backward_frames_reversed))
            chunk_frames = backward_frames_reversed[chunk_start:chunk_end]
            
            # Add seed frame at the beginning for reference
            chunk_with_seed = [self.frame_names[self.seed_frame_idx]] + chunk_frames
            
            if debug:
                original_indices = [self.seed_frame_idx - 1 - chunk_start - i for i in range(len(chunk_frames))]
                print(f"  Backward chunk: frames {min(original_indices)} to {max(original_indices)}")
            
            chunk_results = self._process_chunk(chunk_with_seed, seed_results, is_forward=False, debug=debug)
            
            # Remove seed frame from results (already have it)
            chunk_results.pop(self.seed_frame_idx, None)
            backward_results.update(chunk_results)
        
        return backward_results

    def _process_chunk(self, chunk_frames, reference_results, is_forward=True, debug=True):
        """Process a chunk of frames"""
        chunk_results = {}
        temp_dir = f"temp_{'forward' if is_forward else 'backward'}"
        
        try:
            # Create temp directory
            temp_dir_path = self._create_temp_video_dir(chunk_frames, temp_dir)
            chunk_state = self.predictor.init_state(video_path=temp_dir_path)
            
            # Get reference masks from seed frame
            seed_masks = reference_results.get(self.seed_frame_idx, {})
            
            # Process each object
            for obj_id, reference_mask in seed_masks.items():
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    # Generate points from reference mask
                    points, labels = self._generate_robust_points_from_mask(reference_mask)
                    if points is None:
                        continue
                    
                    # Add prompts to frame 0 (seed frame in temp directory)
                    _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=chunk_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
                    
                    del mask_logits
                    cleanup_memory()
                    
                    # Propagate through chunk
                    for frame_idx, prop_obj_ids, prop_mask_logits in self.predictor.propagate_in_video(chunk_state):
                        # Skip seed frame (frame 0)
                        if frame_idx == 0:
                            continue
                        
                        # Map local frame index to global frame index
                        global_frame_idx = self._map_local_to_global_index(
                            frame_idx, chunk_frames, is_forward
                        )
                        
                        if global_frame_idx is None:
                            continue
                        
                        for i, prop_obj_id in enumerate(prop_obj_ids):
                            mask = (prop_mask_logits[i] > 0.0).cpu().numpy()
                            if len(mask.shape) == 3:
                                mask = mask[0]
                            
                            if global_frame_idx not in chunk_results:
                                chunk_results[global_frame_idx] = {}
                            chunk_results[global_frame_idx][prop_obj_id] = mask.copy()
                        
                        del prop_mask_logits
                        cleanup_memory()
                
                except Exception as e:
                    if debug:
                        print(f"    Error processing object {obj_id}: {e}")
                    continue
            
            return chunk_results
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return {}
        finally:
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)
            cleanup_memory()

    def _map_local_to_global_index(self, local_idx, chunk_frames, is_forward):
        """Map local frame index in chunk to global frame index"""
        if local_idx >= len(chunk_frames):
            return None
        
        frame_name = chunk_frames[local_idx]
        
        # Find global index by frame name
        try:
            global_idx = self.frame_names.index(frame_name)
            return global_idx
        except ValueError:
            return None

    def _fill_result_gaps(self, results, debug=True):
        """Fill small gaps in results using interpolation"""
        if not results:
            return
        
        frame_indices = sorted(results.keys())
        if len(frame_indices) < 2:
            return
        
        gaps = []
        for i in range(len(frame_indices) - 1):
            gap_size = frame_indices[i+1] - frame_indices[i] - 1
            if gap_size > 0 and gap_size <= 5:
                gaps.append((frame_indices[i], frame_indices[i+1], gap_size))
        
        if not gaps:
            return
        
        if debug:
            print(f"Filling {len(gaps)} small gaps in results...")
        
        for start_frame, end_frame, gap_size in gaps:
            start_masks = results[start_frame]
            end_masks = results[end_frame]
            
            common_objects = set(start_masks.keys()) & set(end_masks.keys())
            
            for obj_id in common_objects:
                start_mask = start_masks[obj_id]
                end_mask = end_masks[obj_id]
                
                for gap_frame in range(start_frame + 1, end_frame):
                    ratio = (gap_frame - start_frame) / (end_frame - start_frame)
                    
                    if ratio < 0.5:
                        interpolated_mask = start_mask.copy()
                    else:
                        interpolated_mask = end_mask.copy()
                    
                    if gap_frame not in results:
                        results[gap_frame] = {}
                    results[gap_frame][obj_id] = interpolated_mask

    def create_analysis_video(self, results, output_path, fps=30, alpha=0.5):
        """Create analysis video with plots and metrics - FIXED"""
        if not results:
            print("No results to analyze!")
            return
        
        print("Creating analysis video...")
        
        # Get color map and object names
        cmap = plt.get_cmap("tab10")
        object_names = getattr(self, 'object_names', {})
        
        # Collect time series data
        time_series_data = {}
        max_frame_idx = max(results.keys())
        
        for obj_id in set(obj_id for frame in results.values() for obj_id in frame.keys()):
            time_series_data[obj_id] = {
                'frames': [],
                'centroids': [],
                'areas': [],
                'colors_rgb': [],
                'plot_color': cmap(obj_id % 10)[:3]
            }
        
        # Calculate metrics for all frames
        for frame_idx in sorted(results.keys()):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            
            for obj_id, mask in results[frame_idx].items():
                box = self._compute_box_from_mask(mask)
                metrics = self._compute_metrics(mask, box, frame)
                if metrics is None:
                    continue
                    
                data = time_series_data[obj_id]
                data['frames'].append(frame_idx)
                data['centroids'].append((metrics['seg_centroid_x'], metrics['seg_centroid_y']))
                data['areas'].append(metrics['surface_area'])
                data['colors_rgb'].append([metrics['mean_color_r'], 
                                         metrics['mean_color_g'], 
                                         metrics['mean_color_b']])
        
        # Calculate derived metrics
        window_size = 10
        for obj_id in time_series_data:
            data = time_series_data[obj_id]
            
            if not data['frames']:  # Skip if no data
                continue
            
            # Convert to numpy arrays
            centroids = np.array(data['centroids'])
            if len(centroids) > 1:
                data['movement'] = np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))
                data['movement'] = np.insert(data['movement'], 0, 0)
            else:
                data['movement'] = np.array([0])
            
            colors = np.array(data['colors_rgb'])
            if len(colors) > 1:
                data['color_change'] = np.linalg.norm(np.diff(colors, axis=0), axis=1)
                data['color_change'] = np.insert(data['color_change'], 0, 0)
            else:
                data['color_change'] = np.array([0])
            
            # Calculate moving averages
            data['area_ma'] = np.convolve(data['areas'], 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
            data['movement_ma'] = np.convolve(data['movement'],
                                             np.ones(window_size)/window_size,
                                             mode='same')
            data['color_ma'] = np.convolve(data['color_change'],
                                          np.ones(window_size)/window_size,
                                          mode='same')
        
        # Video setup
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]
        
        # Layout calculation
        n_objects = len([obj_id for obj_id in time_series_data if time_series_data[obj_id]['frames']])
        if n_objects == 0:
            print("No valid objects for analysis video")
            return
        
        side_plot_height = height // max(n_objects, 1)
        side_plot_width = width // 3
        bottom_plot_height = height // 4
        
        # Total output dimensions
        out_width = width + (2 * side_plot_width)
        out_height = height + bottom_plot_height
        
        # Video position in output frame
        video_x = side_plot_width
        video_y = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        for frame_idx in tqdm(range(len(self.frame_names)), desc="Creating analysis video"):
            # Create output canvas
            output_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            # Create masked overlay
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            overlay = frame.copy()
            
            # Store mask centroids for connection lines
            centroids = {}
            
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    if mask.shape != (height, width):
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), 
                                            interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue
                    
                    # Calculate centroid for connection lines
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        centroids[obj_id] = (cx + video_x, cy + video_y)
                    
                    color = np.array(cmap(obj_id % 10)[:3]) * 255
                    color_mask = np.zeros_like(overlay)
                    for c in range(3):
                        color_mask[:, :, c][mask] = color[c]
                    
                    blend_mask = np.zeros_like(overlay)
                    cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0, blend_mask)
                    overlay[mask] = blend_mask[mask]
            
            # Place video in center
            output_frame[video_y:video_y+height, video_x:video_x+width] = overlay
            
            # Create and place plots for each object
            plot_idx = 0
            for obj_id, data in time_series_data.items():
                if not data['frames']:  # Skip empty data
                    continue
                
                plot_color = data['plot_color']
                color_rgb = tuple(int(c * 255) for c in plot_color)
                
                # Get object name
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                
                # Left plot (Movement)
                try:
                    fig_left = Figure(figsize=(side_plot_width/100, side_plot_height/100), dpi=100)
                    ax_left = fig_left.add_subplot(111)
                    ax_left.plot(data['frames'], data['movement'], color=plot_color, alpha=0.5)
                    ax_left.plot(data['frames'], data['movement_ma'], color=plot_color, linewidth=2)
                    ax_left.set_xlim(0, max_frame_idx)
                    ax_left.axvline(frame_idx, color='k', linestyle='--', alpha=0.5)
                    ax_left.set_title(f'Movement ({obj_name})')
                    fig_left.tight_layout()
                    
                    canvas = FigureCanvasAgg(fig_left)
                    canvas.draw()
                    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
                    
                    y_offset = plot_idx * side_plot_height
                    if y_offset + side_plot_height <= out_height:
                        output_frame[y_offset:y_offset+side_plot_height, :side_plot_width] = plot_img
                    
                    plt.close(fig_left)
                except Exception as e:
                    print(f"Error creating left plot for object {obj_id}: {e}")
                
                plot_idx += 1
            
            out.write(output_frame)
        
        out.release()
        print(f"Analysis video saved to: {output_path}")

    def _compute_box_from_mask(self, mask):
        """Compute bounding box from mask"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask = mask.astype(bool)
        
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
            
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        padding = 10
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
        object_names = getattr(self, 'object_names', {})
        
        def get_object_name(obj_id):
            return object_names.get(obj_id, f"Object_{obj_id}")
        
        if show_original:
            out_width = width * 2
        else:
            out_width = width
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
    
        print("Saving video...")
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
                        
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            name = get_object_name(obj_id)
                            
                            # Add background for better text visibility
                            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(overlay, (cx - text_size[0]//2 - 5, cy + 5), 
                                        (cx + text_size[0]//2 + 5, cy + 25), (0, 0, 0), -1)
                            
                            cv2.putText(overlay, name, (cx - text_size[0]//2, cy + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
        object_names = getattr(self, 'object_names', {})
        
        def get_object_identifier(obj_id):
            return object_names.get(obj_id, f"Object_{obj_id}")
        
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
                    
                    obj_identifier = get_object_identifier(obj_id)
                    
                    metrics.update({
                        'frame': frame_idx,
                        'frame_name': frame_name,
                        'object_id': obj_identifier,
                        'object_name': obj_identifier,
                        'numeric_object_id': obj_id,
                        'is_named_object': obj_id in object_names,
                        'timestamp_seconds': frame_idx / 30.0
                    })
                    
                    metrics_data.append(metrics)
                    
            except Exception as e:
                continue
        
        if not metrics_data:
            print("No valid metrics data collected")
            return
            
        df = pd.DataFrame(metrics_data)
        
        # Calculate deltas and velocities
        df['delta_centroid_x'] = df.groupby('object_id')['seg_centroid_x'].diff()
        df['delta_centroid_y'] = df.groupby('object_id')['seg_centroid_y'].diff()
        df['delta_area'] = df.groupby('object_id')['surface_area'].diff()
        
        df['velocity_x'] = df['delta_centroid_x'].fillna(0)
        df['velocity_y'] = df['delta_centroid_y'].fillna(0) 
        df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
        
        df['cumulative_distance'] = df.groupby('object_id')['velocity_magnitude'].cumsum()
        
        df.to_csv(csv_path, index=False)
        print(f"Saved time series metrics to: {csv_path}")

    def _save_coco_annotations(self, json_path):
        """Save annotations in COCO format"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        object_names = getattr(self, 'object_names', {})
        
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
            "categories": []
        }
        
        # Create categories
        unique_objects = set()
        for frame_results in self.results.values():
            unique_objects.update(frame_results.keys())
        
        for obj_id in sorted(unique_objects):
            obj_name = object_names.get(obj_id, f"Object_{obj_id}")
            coco_data["categories"].append({
                "supercategory": "object",
                "id": obj_id,
                "name": obj_name
            })
        
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
                            "category_id": obj_id,
                            "frame_number": frame_idx
                        })
                        annotation_id += 1
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
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
    object_names = {}
    current_obj_id = 1
    
    temp_dir = "temp_select"
    chunk_state = None
    
    if processor is not None:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(frame_path, frame)
            
            chunk_state = processor.predictor.init_state(video_path=temp_dir)
            if chunk_state is None:
                raise ValueError("Failed to initialize chunk state")
            
        except Exception as e:
            print(f"Error initializing processor: {str(e)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            processor = None
    
    def get_object_name(obj_id):
        if obj_id in object_names:
            return f"{obj_id}:{object_names[obj_id]}"
        else:
            return str(obj_id)
    
    def draw_point(img, point, obj_id, label):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
        
        display_name = get_object_name(obj_id)
        cv2.putText(img, display_name, 
                   (int(point[0] + 5), int(point[1] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def redraw_all_points():
        display = frame.copy()
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        
        height, width = display.shape[:2]
        
        # Create semi-transparent overlay for instructions
        overlay = display.copy()
        instructions_height = 200
        cv2.rectangle(overlay, (10, height - instructions_height - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        instructions = [
            "KEYBOARD SHORTCUTS:",
            "Left Click: Add positive point (+)",
            "Right Click: Add negative point (-)",
            "R: Reset current object",
            "N: Next object  P: Previous object",
            "C: Name current object",
            "T: Test/preview mask",
            "Enter: Finish  Q: Quit"
        ]
        
        y_start = height - instructions_height
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            font_scale = 0.6 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            
            cv2.putText(display, instruction, (20, y_start + (i * 22)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        current_obj_name = get_object_name(current_obj_id)
        obj_info = f"Current Object: {current_obj_name}"
        cv2.putText(display, obj_info, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if current_obj_id in points_dict:
            pos_count = sum(1 for l in labels_dict[current_obj_id] if l == 1)
            neg_count = sum(1 for l in labels_dict[current_obj_id] if l == 0)
            count_info = f"Points: +{pos_count} -{neg_count}"
            cv2.putText(display, count_info, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return display
    
    def name_current_object():
        import tkinter as tk
        from tkinter import simpledialog
        
        root = tk.Tk()
        root.withdraw()
        
        current_name = object_names.get(current_obj_id, f"Object_{current_obj_id}")
        name = simpledialog.askstring("Object Name", 
                                     f"Enter name for object {current_obj_id}:",
                                     initialvalue=current_name)
        root.destroy()
        
        if name and name.strip():
            object_names[current_obj_id] = name.strip()
            print(f"Object {current_obj_id} named: {object_names[current_obj_id]}")
            
            nonlocal img_display
            img_display = redraw_all_points()
        
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
                    mask = cv2.resize(mask.astype(np.float32), 
                                    (width, height),
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
                
                title = f"Preview: {get_object_name(current_obj_id)}"
                cv2.putText(preview, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
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
                if current_obj_id not in object_names:
                    object_names[current_obj_id] = f"Object_{current_obj_id}"
            
            points_dict[current_obj_id].append([x, y])
            label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            labels_dict[current_obj_id].append(label)
            
            img_display = redraw_all_points()
            obj_name = get_object_name(current_obj_id)
            print(f"Added {'positive' if label == 1 else 'negative'} point for {obj_name}")
    
    img_display = redraw_all_points()
    cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Points', click_handler)
    
    print("\nControls:")
    print("- Left click: add positive point (green)")
    print("- Right click: add negative point (red)")
    print("- Press 'r' to reset current object")
    print("- Press 'n' for next object")
    print("- Press 'p' for previous object")
    print("- Press 'c' to name current object")
    print("- Press 't' to test mask")
    print("- Press Enter to finish")
    print("- Press 'q' to quit")
    
    while True:
        cv2.imshow('Select Points', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            if current_obj_id in points_dict:
                points_dict[current_obj_id] = []
                labels_dict[current_obj_id] = []
                img_display = redraw_all_points()
                obj_name = get_object_name(current_obj_id)
                print(f"Reset points for {obj_name}")
        
        elif key == ord('n'):
            current_obj_id += 1
            obj_name = get_object_name(current_obj_id)
            print(f"Now selecting {obj_name}")
            img_display = redraw_all_points()
        
        elif key == ord('p'):
            if current_obj_id > 1:
                current_obj_id -= 1
                obj_name = get_object_name(current_obj_id)
                print(f"Now selecting {obj_name}")
                img_display = redraw_all_points()
        
        elif key == ord('c'):
            name_current_object()
        
        elif key == ord('t'):
            test_mask()
        
        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return points_dict, labels_dict, object_names if points_dict else (None, None, None)
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, None, None
    
    return points_dict, labels_dict, object_names

def create_finetuning_setup():
    """Create fine-tuning setup files - FIXED for encoding issues"""
    try:
        # Training config template
        config_content = """# SAM2 Fine-tuning Configuration
model:
  type: "sam2_hiera_l"
  checkpoint: "checkpoints/sam2.1_hiera_large.pt" 
  freeze_image_encoder: false
  
dataset:
  name: "custom_objects"
  annotation_dirs: ["./video1_frames", "./video2_frames"]  # Update with your paths
  object_names: ["Object_1", "Object_2"]  # Update with your object names
  train_split: 0.8
  val_split: 0.2
  image_size: [1024, 1024]
  
training:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 50
  weight_decay: 1e-4
  
output:
  save_dir: "./sam2_finetuned"
  save_every: 10
  eval_every: 5
"""

        # Training script template
        training_script = '''#!/usr/bin/env python3
"""
SAM2 Fine-tuning Training Script - Simplified
"""

import os
import yaml
import torch

def main():
    print("SAM2 Fine-tuning Training")
    print("=" * 40)
    
    # Check if config exists
    if not os.path.exists("training_config.yaml"):
        print("Error: training_config.yaml not found")
        return
    
    print("Loading configuration...")
    with open("training_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully!")
    print(f"Model: {config['model']['type']}")
    print(f"Objects: {config['dataset']['object_names']}")
    print(f"Annotation dirs: {config['dataset']['annotation_dirs']}")
    
    # Check if SAM2 is available
    try:
        from sam2.build_sam import build_sam2_video_predictor
        print("SAM2 is available!")
    except ImportError:
        print("Error: SAM2 not found. Please install SAM2 first.")
        print("git clone https://github.com/facebookresearch/segment-anything-2.git")
        print("cd segment-anything-2")
        print("pip install -e .")
        return
    
    print("\\nTo implement actual training:")
    print("1. Create custom dataset class")
    print("2. Implement training loop")
    print("3. Add loss functions")
    print("4. Set up validation")
    
    print("\\nFor now, this is a template. Full implementation requires:")
    print("- Custom dataset loader for your annotations")
    print("- Training loop with SAM2 model")
    print("- Loss computation and backpropagation")

if __name__ == "__main__":
    main()
'''

        # Write files with proper encoding
        with open("training_config.yaml", "w", encoding='utf-8') as f:
            f.write(config_content)
        print("✅ Created: training_config.yaml")
        
        with open("train_sam2.py", "w", encoding='utf-8') as f:
            f.write(training_script)
        print("✅ Created: train_sam2.py")
        
        print("\n🎯 Fine-tuning setup complete!")
        print("\nNext steps:")
        print("1. Update training_config.yaml with your annotation directories and object names")
        print("2. Ensure you have multiple annotated videos")
        print("3. Run: python train_sam2.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating fine-tuning setup: {e}")
        return False

class VideoAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAM2 Video Analysis - Complete Fixed Version")
        self.root.geometry("700x750")
        self.root.minsize(700, 750)
        
        # Initialize SAM2
        self.device = setup_device()
        self.predictor = None
        self.init_sam2()
        
        self.setup_gui()
        
    def init_sam2(self):
        """Initialize SAM2 predictor"""
        try:
            # Update these paths to your SAM2 installation
            sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if not os.path.exists(sam2_checkpoint):
                messagebox.showwarning("SAM2 Setup", 
                    f"SAM2 checkpoint not found at: {sam2_checkpoint}\n"
                    "Please update the path in the script or download SAM2 checkpoints.")
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
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="SAM2 Video Analysis - Complete Fixed Version", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Folder selection
        folder_frame = tk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(folder_frame, text="Select folder containing videos:", 
                font=("Arial", 9)).pack(anchor=tk.W)
        
        folder_input_frame = tk.Frame(folder_frame)
        folder_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.folder_var = tk.StringVar()
        folder_entry = tk.Entry(folder_input_frame, textvariable=self.folder_var, 
                               width=40, state='readonly')
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(folder_input_frame, text="Browse", 
                 command=self.select_folder).pack(side=tk.RIGHT)
        
        # Video list
        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))
        
        tk.Label(list_frame, text="Videos found:", 
                font=("Arial", 9)).pack(anchor=tk.W)
        
        listbox_frame = tk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.video_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, height=6)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.video_listbox.yview)
        
        # Processing options - ENHANCED
        options_frame = tk.LabelFrame(main_frame, text="Processing Options", font=("Arial", 9, "bold"))
        options_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Chunk size setting - NEW
        chunk_frame = tk.Frame(options_frame)
        chunk_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(chunk_frame, text="Chunk Size (frames):").pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value="500")
        chunk_spin = tk.Spinbox(chunk_frame, from_=100, to=2000, increment=100, 
                               textvariable=self.chunk_size_var, width=10)
        chunk_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(chunk_frame, text="(smaller = less memory, slower processing)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Interactive correction option
        self.interactive_correction = tk.BooleanVar(value=True)
        correction_cb = tk.Checkbutton(options_frame, 
                                      text="🎯 Interactive mask correction",
                                      variable=self.interactive_correction)
        correction_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Analysis video option
        self.create_analysis_video = tk.BooleanVar(value=True)
        analysis_cb = tk.Checkbutton(options_frame, 
                                    text="📊 Create analysis video with plots",
                                    variable=self.create_analysis_video)
        analysis_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Fine-tuning section
        finetuning_frame = tk.LabelFrame(main_frame, text="🧠 Fine-tuning", font=("Arial", 9, "bold"))
        finetuning_frame.pack(fill=tk.X, pady=(10, 10))
        
        tk.Button(finetuning_frame, text="🔧 Setup Fine-tuning Environment", 
                 command=self.setup_finetuning, bg="#FF5722", fg="white",
                 font=("Arial", 9)).pack(fill=tk.X, padx=5, pady=2)
        
        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(15, 10))
        
        self.process_button = tk.Button(process_frame, text="🎬 Process Selected Video", 
                                       command=self.process_video, bg="#4CAF50", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select a folder and video to begin")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 8), wraplength=650)
        status_label.pack(pady=(5, 0))
    
    def setup_finetuning(self):
        """Setup fine-tuning environment - FIXED"""
        try:
            success = create_finetuning_setup()
            if success:
                messagebox.showinfo("Setup Complete", 
                    "Fine-tuning environment setup complete!\n\n"
                    "Files created:\n"
                    "- training_config.yaml\n"
                    "- train_sam2.py\n\n"
                    "Next: Update training_config.yaml with your paths and run training")
            else:
                messagebox.showerror("Setup Failed", "Failed to create fine-tuning setup")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup fine-tuning: {str(e)}")
    
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
            self.video_listbox.select_set(0)
            self.status_var.set(f"Found {len(videos)} video(s)")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number_with_preview(self, frames_dir, total_frames):
        """Get frame number with preview functionality - NEW"""
        suggested_frame = total_frames // 2
        
        while True:
            frame_num = simpledialog.askinteger(
                "Reference Frame Selection",
                f"Select frame for object annotation (0-{total_frames-1}):\n\n"
                f"💡 Choose a frame where objects are clearly visible\n"
                f"🔄 Processing will propagate forward AND backward from this frame\n"
                f"📍 Suggested: Frame {suggested_frame} (middle of video)\n\n"
                f"Enter frame number (or -1 to preview suggested frame):",
                minvalue=-1,
                maxvalue=total_frames-1,
                initialvalue=suggested_frame
            )
            
            if frame_num is None:  # User cancelled
                return None
            
            if frame_num == -1:  # Preview requested
                if show_frame_preview(frames_dir, suggested_frame, total_frames):
                    continue  # Show dialog again
                else:
                    return None
            
            # Show preview of selected frame
            if show_frame_preview(frames_dir, frame_num, total_frames):
                # Ask for confirmation
                confirm = messagebox.askyesno("Confirm Frame Selection", 
                    f"Use frame {frame_num} as reference frame?\n\n"
                    "This frame will be used for object annotation and\n"
                    "processing will propagate both forward and backward from here.")
                
                if confirm:
                    return frame_num
                # If not confirmed, loop back to frame selection
            else:
                return None  # Error showing preview
    
    def process_video(self):
        """Process the selected video - ENHANCED"""
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
            
            # Get frame number with preview - ENHANCED
            frame_num = self.get_frame_number_with_preview(frames_dir, num_frames)
            if frame_num is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Get chunk size from UI - NEW
            try:
                chunk_size = int(self.chunk_size_var.get())
            except ValueError:
                chunk_size = 500
                self.chunk_size_var.set("500")
            
            # Initialize processor with seed frame and custom chunk size
            self.status_var.set("Initializing processor...")
            self.root.update()
            
            processor = VideoChunkProcessor(
                predictor=self.predictor, 
                video_dir=frames_dir, 
                chunk_size=chunk_size,  # Use UI value
                overlap_frames=30,
                interactive_correction=self.interactive_correction.get(),
                seed_frame_idx=frame_num
            )
            
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
                f"Frame {frame_num} will open for annotation.\n\n"
                "🎯 All controls are shown on the frame\n"
                "✓ Left click: positive points\n"
                "✓ Right click: negative points\n"
                "✓ Press 'C' to name objects\n"
                "✓ Press 'T' to preview masks\n\n"
                f"📍 Frame {frame_num} is your reference point\n"
                "🔄 Processing goes both directions from here")
            
            points_dict, labels_dict, object_names = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Process video with bidirectional propagation
            self.status_var.set(f"🔄 Processing video (bidirectional from frame {frame_num}, chunk size {chunk_size})...")
            self.root.update()
            
            results = processor.process_video(points_dict, labels_dict)
            
            if results:
                processor.results = results
                processor.object_names = object_names
                
                # Save basic results
                self.status_var.set("Saving results...")
                self.root.update()
                
                output_path = os.path.join(frames_dir, "output_masked.mp4")
                processor.save_results(
                    output_path=output_path,
                    fps=fps,
                    show_original=True,
                    alpha=0.5
                )
                
                # Create analysis video if requested - FIXED
                if self.create_analysis_video.get():
                    self.status_var.set("Creating analysis video...")
                    self.root.update()
                    
                    analysis_output = os.path.join(frames_dir, "analysis_video.mp4")
                    try:
                        processor.create_analysis_video(
                            results=results,
                            output_path=analysis_output,
                            fps=fps
                        )
                        analysis_created = True
                    except Exception as e:
                        print(f"Error creating analysis video: {e}")
                        analysis_created = False
                else:
                    analysis_created = False
                
                self.status_var.set("Processing completed!")
                
                # Create success message
                named_objects = [name for name in object_names.values()]
                total_objects = len(object_names)
                objects_summary = "\n".join([f"  • {name}" for name in named_objects])
                
                success_msg = f"""🎉 Processing Complete!

Reference Frame: {frame_num}
Chunk Size: {chunk_size} frames
Results saved in: {frames_dir}

📁 Generated Files:
• output_masked.mp4 - Video with overlays
• segmentation_coco.json - Annotations
• time_series_metrics.csv - Movement data"""

                if analysis_created:
                    success_msg += "\n• analysis_video.mp4 - Analysis with plots"

                success_msg += f"""

📊 Processed Objects ({total_objects}):
{objects_summary}

✅ Bidirectional processing completed successfully!"""
                
                messagebox.showinfo("Success", success_msg)
                
            else:
                messagebox.showerror("Error", "Video processing failed")
                self.status_var.set("Processing failed")
        
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting SAM2 Video Analysis - Complete Fixed Version")
    print("=" * 50)
    print("✅ Fixed Issues:")
    print("  • Frame preview when selecting reference frame")
    print("  • Bidirectional propagation (backward processing)")
    print("  • Analysis video creation with proper object names")
    print("  • Fine-tuning setup with encoding fixes")
    print("  • Configurable chunk size in UI")
    print("  • Combined into single comprehensive script")
    print("\nFeatures:")
    print("  • Interactive video annotation with SAM2")
    print("  • Bidirectional processing from any reference frame")
    print("  • Analysis videos with movement/area/color plots")
    print("  • Custom object naming")
    print("  • Configurable processing parameters")
    print("  • Fine-tuning workflow setup")
    print("\nRequirements:")
    print("  • SAM2 installed and checkpoints downloaded")
    print("  • FFmpeg for video processing")
    print("  • Python packages: opencv-python, torch, matplotlib, pandas, tqdm")
    print("\nStarting application...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()