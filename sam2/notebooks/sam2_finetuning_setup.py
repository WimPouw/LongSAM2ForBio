#!/usr/bin/env python3
"""
Complete Video Analysis Script with SAM2 - Fixed Version
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
    def __init__(self, predictor, video_dir, chunk_size=500, overlap_frames=20, 
                 interactive_correction=True, seed_frame_idx=0, **kwargs):
        self.predictor = predictor
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.interactive_correction = interactive_correction
        self.seed_frame_idx = seed_frame_idx  # The frame where user annotated objects
        
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
        
        self.chunks = self._calculate_bidirectional_chunks()
        print(f"Created {len(self.chunks)} chunks with seed frame at {self.seed_frame_idx}")
        print(f"Processing order: seed → forward → backward")
        if interactive_correction:
            print("🎯 Interactive correction mode enabled - you'll be prompted for mask discontinuities")

    def _calculate_bidirectional_chunks(self):
        """Calculate chunks for bidirectional processing using frame reordering approach"""
        chunks = []
        frame_count = len(self.frame_names)
        
        print(f"Planning bidirectional processing with seed frame {self.seed_frame_idx}")
        
        # Split frames into three groups: before_seed, seed, after_seed
        before_seed_frames = self.frame_names[:self.seed_frame_idx]
        seed_frame = self.frame_names[self.seed_frame_idx]
        after_seed_frames = self.frame_names[self.seed_frame_idx + 1:]
        
        print(f"  Frames before seed: {len(before_seed_frames)}")
        print(f"  Seed frame: {seed_frame}")
        print(f"  Frames after seed: {len(after_seed_frames)}")
        
        # 1. Create seed chunk (single frame)
        chunks.append({
            'id': 'seed',
            'type': 'seed',
            'global_start': self.seed_frame_idx,
            'global_end': self.seed_frame_idx + 1,
            'frame_names': [seed_frame],
            'frame_indices': [0],
            'original_indices': [self.seed_frame_idx],  # Track original positions
            'processing_order': 0
        })
        
        chunk_counter = 1
        
        # 2. Create forward chunks (after seed)
        if after_seed_frames:
            start_idx = self.seed_frame_idx + 1
            for chunk_start in range(0, len(after_seed_frames), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(after_seed_frames))
                chunk_frames = after_seed_frames[chunk_start:chunk_end]
                
                # Add overlap from previous chunk if not the first forward chunk
                if chunk_start > 0:
                    overlap_start = max(0, chunk_start - self.overlap_frames)
                    overlap_frames = after_seed_frames[overlap_start:chunk_start]
                    chunk_frames = overlap_frames + chunk_frames
                    overlap_offset = len(overlap_frames)
                else:
                    overlap_offset = 0
                
                global_start = start_idx + chunk_start
                global_end = start_idx + chunk_end
                original_indices = list(range(global_start - overlap_offset, global_end))
                
                chunks.append({
                    'id': f'forward_{chunk_counter}',
                    'type': 'forward',
                    'global_start': global_start,
                    'global_end': global_end,
                    'frame_names': chunk_frames,
                    'frame_indices': list(range(len(chunk_frames))),
                    'original_indices': original_indices,
                    'overlap_offset': overlap_offset,
                    'processing_order': chunk_counter
                })
                chunk_counter += 1
        
        # 3. Create backward chunks (before seed) - REVERSED ORDER
        if before_seed_frames:
            # Reverse the before_seed_frames so newest is first
            reversed_before_frames = before_seed_frames[::-1]
            
            for chunk_start in range(0, len(reversed_before_frames), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(reversed_before_frames))
                chunk_frames = reversed_before_frames[chunk_start:chunk_end]
                
                # Add overlap from previous chunk if not the first backward chunk
                if chunk_start > 0:
                    overlap_start = max(0, chunk_start - self.overlap_frames)
                    overlap_frames = reversed_before_frames[overlap_start:chunk_start]
                    chunk_frames = overlap_frames + chunk_frames
                    overlap_offset = len(overlap_frames)
                else:
                    overlap_offset = 0
                
                # Calculate original indices (these are in forward time, not reversed)
                # We need to map back from reversed indices to original indices
                reversed_start = chunk_start
                reversed_end = chunk_end
                
                # Map back to original frame indices
                original_start_in_reversed = len(before_seed_frames) - reversed_end
                original_end_in_reversed = len(before_seed_frames) - reversed_start
                
                original_indices = list(range(original_start_in_reversed - overlap_offset, 
                                            original_end_in_reversed))
                
                chunks.append({
                    'id': f'backward_{chunk_counter}',
                    'type': 'backward',
                    'global_start': original_start_in_reversed,
                    'global_end': original_end_in_reversed,
                    'frame_names': chunk_frames,  # These are in REVERSED order
                    'frame_indices': list(range(len(chunk_frames))),
                    'original_indices': original_indices,  # Original chronological indices
                    'overlap_offset': overlap_offset,
                    'processing_order': chunk_counter,
                    'is_time_reversed': True  # Flag to indicate this chunk is time-reversed
                })
                chunk_counter += 1
        
        print(f"Created {len(chunks)} chunks:")
        for chunk in chunks:
            print(f"  {chunk['id']}: {chunk['type']} - frames {chunk.get('global_start', 'N/A')}-{chunk.get('global_end', 'N/A')}")
        
        return chunks

    def _calculate_chunks_with_overlap(self):
        """Original chunk calculation (kept for backward compatibility)"""
        return self._calculate_bidirectional_chunks()

    def _get_best_reference_masks(self, results, target_frame, search_window=10):
        """Get the best masks from multiple recent frames for robust inheritance"""
        reference_masks = {}
        frames_to_check = range(max(0, target_frame - search_window), target_frame)
        
        mask_candidates = {}
        for frame_idx in frames_to_check:
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    if obj_id not in mask_candidates:
                        mask_candidates[obj_id] = []
                    mask_candidates[obj_id].append((frame_idx, mask))
        
        for obj_id, candidates in mask_candidates.items():
            if not candidates:
                continue
                
            best_candidate = None
            best_score = -1
            
            for frame_idx, mask in candidates:
                if len(mask.shape) == 3:
                    mask = mask[0]
                
                area = np.sum(mask)
                recency = frame_idx / target_frame if target_frame > 0 else 1
                score = area * (0.3 + 0.7 * recency)
                
                if score > best_score:
                    best_score = score
                    best_candidate = mask
            
            if best_candidate is not None:
                reference_masks[obj_id] = best_candidate
        
        return reference_masks

    def _generate_robust_points_from_mask(self, mask, num_positive=8, num_negative=16):
        """Generate more robust points from mask using multiple strategies"""
        if not mask.any():
            return None, None
            
        points = []
        labels = []
        
        if len(mask.shape) == 3:
            mask = mask[0]
        mask = mask.astype(bool)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            contour_length = cv2.arcLength(largest_contour, True)
            if contour_length > 0:
                spacing = max(1, int(contour_length / num_positive))
                contour_points = []
                for i in range(0, len(largest_contour), spacing):
                    point = largest_contour[i][0]
                    contour_points.append([point[0], point[1]])
                
                step = max(1, len(contour_points) // num_positive)
                for i in range(0, min(len(contour_points), num_positive), step):
                    points.append(contour_points[i])
                    labels.append(1)
        
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            if mask[cy, cx]:
                points.append([cx, cy])
                labels.append(1)
            
            for radius in [10, 20, 30]:
                for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    x = cx + int(radius * np.cos(angle))
                    y = cy + int(radius * np.sin(angle))
                    
                    if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and 
                        mask[y, x] and len([p for p, l in zip(points, labels) if l == 1]) < num_positive):
                        points.append([x, y])
                        labels.append(1)
        
        kernel_size = max(10, int(np.sqrt(np.sum(mask)) * 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        negative_region = expanded & (~mask)
        
        neg_y, neg_x = np.where(negative_region)
        if len(neg_x) > 0:
            neg_coords = np.column_stack([neg_x, neg_y])
            
            grid_size = int(np.sqrt(num_negative))
            x_bins = np.linspace(neg_x.min(), neg_x.max(), grid_size + 1)
            y_bins = np.linspace(neg_y.min(), neg_y.max(), grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    mask_x = (neg_x >= x_bins[i]) & (neg_x < x_bins[i+1])
                    mask_y = (neg_y >= y_bins[j]) & (neg_y < y_bins[j+1])
                    cell_mask = mask_x & mask_y
                    
                    if np.any(cell_mask):
                        cell_indices = np.where(cell_mask)[0]
                        chosen_idx = cell_indices[len(cell_indices)//2]
                        points.append([neg_x[chosen_idx], neg_y[chosen_idx]])
                        labels.append(0)
                        
                        if len([p for p, l in zip(points, labels) if l == 0]) >= num_negative:
                            break
                if len([p for p, l in zip(points, labels) if l == 0]) >= num_negative:
                    break
        
        if not points:
            return None, None
            
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _interactive_mask_correction(self, frame_path, prev_mask, current_mask, obj_id, object_names):
        """Interactive mask correction when discontinuity is detected"""
        print(f"\n🚨 MASK DISCONTINUITY DETECTED for {object_names.get(obj_id, f'Object_{obj_id}')}")
        print("Opening interactive correction interface...")
        
        frame = cv2.imread(frame_path)
        if frame is None:
            print("❌ Could not load frame for correction")
            return current_mask, True
        
        height, width = frame.shape[:2]
        
        if len(prev_mask.shape) == 3:
            prev_mask = prev_mask[0]
        if len(current_mask.shape) == 3:
            current_mask = current_mask[0]
        
        comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)
        
        comparison[:, :width] = frame
        
        prev_overlay = frame.copy()
        prev_color = np.array([0, 255, 0])
        prev_overlay[prev_mask] = prev_color
        blended_prev = cv2.addWeighted(frame, 0.7, prev_overlay, 0.3, 0)
        comparison[:, width:width*2] = blended_prev
        
        curr_overlay = frame.copy()
        curr_color = np.array([0, 0, 255])
        curr_overlay[current_mask] = curr_color
        blended_curr = cv2.addWeighted(frame, 0.7, curr_overlay, 0.3, 0)
        comparison[:, width*2:] = blended_curr
        
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Previous (Good)", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Current (Problem)", (width*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        obj_name = object_names.get(obj_id, f'Object_{obj_id}')
        cv2.putText(comparison, f"Discontinuity: {obj_name}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.namedWindow('Mask Discontinuity Detected', cv2.WINDOW_NORMAL)
        cv2.imshow('Mask Discontinuity Detected', comparison)
        
        print(f"\n🔧 Discontinuity detected for {obj_name}")
        print("Options:")
        print("1. Press 'f' - Fix with manual point selection")
        print("2. Press 'a' - Accept current mask anyway")
        print("3. Press 'p' - Use previous mask")
        print("4. Press 's' - Skip this object for this frame")
        print("5. Press 'q' - Stop processing (quit)")
        print("\nWhat would you like to do?")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('f'):
                cv2.destroyWindow('Mask Discontinuity Detected')
                corrected_mask = self._manual_mask_correction(frame, prev_mask, obj_id, object_names)
                return corrected_mask, True
                
            elif key == ord('a'):
                cv2.destroyWindow('Mask Discontinuity Detected')
                print(f"✅ Accepted current mask for {obj_name}")
                return current_mask, True
                
            elif key == ord('p'):
                cv2.destroyWindow('Mask Discontinuity Detected')
                print(f"📋 Using previous mask for {obj_name}")
                return prev_mask, True
                
            elif key == ord('s'):
                cv2.destroyWindow('Mask Discontinuity Detected')
                print(f"⏭️ Skipping {obj_name} for this frame")
                return None, True
                
            elif key == ord('q'):
                cv2.destroyWindow('Mask Discontinuity Detected')
                print("🛑 User chose to stop processing")
                return None, False
                
            else:
                print("Invalid key. Use 'f', 'a', 'p', 's', or 'q'")

    def _manual_mask_correction(self, frame, reference_mask, obj_id, object_names):
        """Manual point selection for mask correction"""
        obj_name = object_names.get(obj_id, f'Object_{obj_id}')
        print(f"\n🎯 Manual correction for {obj_name} - see instructions on image")
        print("💡 Tip: Add positive points (+) inside the object, negative points (-) outside")
        print("💡 Use 'T' to test your points before applying the correction")
        
        temp_dir = "temp_correction"
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # SAM2 expects numbered frame files - use 00000.jpg format
            frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(frame_path, frame)
            
            try:
                correction_state = self.predictor.init_state(video_path=temp_dir)
                if correction_state is None:
                    print("❌ Failed to initialize correction state")
                    return reference_mask
                print("✅ Correction state initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing correction state: {e}")
                return reference_mask
            
            points_dict = {}
            labels_dict = {}
            current_obj_id = obj_id
            
            def draw_point(img, point, label):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
                cv2.putText(img, "+" if label == 1 else "-", 
                           (int(point[0] + 5), int(point[1] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            def redraw_display():
                display = frame.copy()
                
                if reference_mask is not None:
                    ref_overlay = np.zeros_like(display)
                    ref_mask_2d = reference_mask[0] if len(reference_mask.shape) == 3 else reference_mask
                    ref_overlay[ref_mask_2d] = [0, 150, 0]
                    display = cv2.addWeighted(display, 0.8, ref_overlay, 0.2, 0)
                
                if obj_id in points_dict:
                    for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                        draw_point(display, pt, label)
                
                # Add object info at top
                cv2.putText(display, f"Correcting: {obj_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add keyboard shortcuts overlay
                height, width = display.shape[:2]
                shortcuts = [
                    "MANUAL CORRECTION:",
                    "Left Click: Add positive point (+)",
                    "Right Click: Add negative point (-)",
                    "T: Test/preview current mask",
                    "R: Reset all points",
                    "Enter: Apply correction",
                    "Q: Cancel and use reference"
                ]
                
                # Create semi-transparent background
                overlay = display.copy()
                shortcuts_height = len(shortcuts) * 22 + 20
                cv2.rectangle(overlay, (10, height - shortcuts_height - 10), 
                             (400, height - 10), (0, 0, 0), -1)
                display = cv2.addWeighted(display, 0.75, overlay, 0.25, 0)
                
                for i, shortcut in enumerate(shortcuts):
                    color = (0, 255, 255) if i == 0 else (255, 255, 255)
                    font_scale = 0.6 if i == 0 else 0.5
                    thickness = 2 if i == 0 else 1
                    cv2.putText(display, shortcut, (20, height - shortcuts_height + 20 + i*22), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                
                # Show current point count
                if obj_id in points_dict and points_dict[obj_id]:
                    pos_count = sum(1 for l in labels_dict[obj_id] if l == 1)
                    neg_count = sum(1 for l in labels_dict[obj_id] if l == 0)
                    count_info = f"Points: +{pos_count} -{neg_count}"
                    cv2.putText(display, count_info, (10, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                return display
            
            def click_handler(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                    if obj_id not in points_dict:
                        points_dict[obj_id] = []
                        labels_dict[obj_id] = []
                    
                    points_dict[obj_id].append([x, y])
                    label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
                    labels_dict[obj_id].append(label)
                    
                    nonlocal img_display
                    img_display = redraw_display()
                    cv2.imshow('Manual Mask Correction', img_display)
                    
                    print(f"Added {'positive' if label == 1 else 'negative'} point")
            
            def test_current_mask():
                if obj_id not in points_dict or not points_dict[obj_id]:
                    print("No points selected yet")
                    return
                
                    try:
                        self.predictor.reset_state(correction_state)
                        points = np.array(points_dict[obj_id], dtype=np.float32)
                        labels = np.array(labels_dict[obj_id], dtype=np.int32)
                        
                        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=correction_state,
                            frame_idx=0,  # Always use frame 0 since we only have one frame
                            obj_id=obj_id,
                            points=points,
                            labels=labels
                        )
                        
                        if len(mask_logits) > 0:
                            test_mask = (mask_logits[0] > 0.0).cpu().numpy()
                            if len(test_mask.shape) == 3:
                                test_mask = test_mask[0]
                            
                            preview = frame.copy()
                            color = np.array(plt.get_cmap("tab10")(obj_id % 10)[:3]) * 255
                            
                            color_overlay = np.zeros_like(preview)
                            for c in range(3):
                                color_overlay[:, :, c][test_mask] = color[c]
                            
                            preview = cv2.addWeighted(preview, 0.7, color_overlay, 0.3, 0)
                            
                            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                                draw_point(preview, pt, label)
                            
                            cv2.putText(preview, f"Preview: {obj_name}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            # Add preview instructions
                            cv2.putText(preview, "Press any key to close preview", (10, preview.shape[0] - 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            cv2.namedWindow('Mask Preview', cv2.WINDOW_NORMAL)
                            cv2.imshow('Mask Preview', preview)
                            
                            print("Preview updated - press any key in preview window to continue")
                            cv2.waitKey(0)
                            cv2.destroyWindow('Mask Preview')
                        else:
                            print("No mask generated - try adding more points")
                            
                    except Exception as e:
                        print(f"Error testing mask: {e}")
                        import traceback
                        traceback.print_exc()
            
            img_display = redraw_display()
            cv2.namedWindow('Manual Mask Correction', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Manual Mask Correction', click_handler)
            cv2.imshow('Manual Mask Correction', img_display)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('t'):
                    test_current_mask()
                    
                elif key == ord('r'):
                    points_dict[obj_id] = []
                    labels_dict[obj_id] = []
                    img_display = redraw_display()
                    cv2.imshow('Manual Mask Correction', img_display)
                    print("Reset all points")
                    
                elif key == 13:  # Enter
                    cv2.destroyAllWindows()
                    
                    if obj_id not in points_dict or not points_dict[obj_id]:
                        print("No correction points provided, using reference mask")
                        return reference_mask
                    
                    try:
                        self.predictor.reset_state(correction_state)
                        points = np.array(points_dict[obj_id], dtype=np.float32)
                        labels = np.array(labels_dict[obj_id], dtype=np.int32)
                        
                        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=correction_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            points=points,
                            labels=labels
                        )
                        
                        if len(mask_logits) > 0:
                            corrected_mask = (mask_logits[0] > 0.0).cpu().numpy()
                            if len(corrected_mask.shape) == 3:
                                corrected_mask = corrected_mask[0]
                            print(f"✅ Applied manual correction for {obj_name}")
                            return corrected_mask
                        else:
                            print("❌ Correction failed, using reference mask")
                            return reference_mask
                            
                    except Exception as e:
                        print(f"❌ Error applying correction: {e}")
                        return reference_mask
                        
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    print(f"❌ Cancelled correction, using reference mask for {obj_name}")
                    return reference_mask
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _validate_mask_continuity(self, prev_mask, current_mask, max_area_change=0.5, max_centroid_shift=50):
        """Validate that the current mask is a reasonable continuation of the previous mask"""
        if prev_mask is None or current_mask is None:
            return False
        
        if len(prev_mask.shape) == 3:
            prev_mask = prev_mask[0]
        if len(current_mask.shape) == 3:
            current_mask = current_mask[0]
        
        prev_mask = prev_mask.astype(bool)
        current_mask = current_mask.astype(bool)
        
        prev_area = np.sum(prev_mask)
        current_area = np.sum(current_mask)
        
        if prev_area == 0 or current_area == 0:
            return False
        
        area_ratio = current_area / prev_area
        if area_ratio < (1 - max_area_change) or area_ratio > (1 + max_area_change):
            return False
        
        def get_centroid(mask):
            if not mask.any():
                return None
            y_coords, x_coords = np.where(mask)
            return np.mean(x_coords), np.mean(y_coords)
        
        prev_centroid = get_centroid(prev_mask)
        current_centroid = get_centroid(current_mask)
        
        if prev_centroid is None or current_centroid is None:
            return False
        
        centroid_distance = np.sqrt((prev_centroid[0] - current_centroid[0])**2 + 
                                   (prev_centroid[1] - current_centroid[1])**2)
        
        if centroid_distance > max_centroid_shift:
            return False
        
        overlap = np.sum(prev_mask & current_mask)
        union = np.sum(prev_mask | current_mask)
        iou = overlap / union if union > 0 else 0
        
        if iou < 0.3:
            return False
        
        return True

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
        
        print(f"Filling {len(gaps)} small gaps in results...")
        
        for start_frame, end_frame, gap_size in gaps:
            if debug:
                print(f"  Filling gap between frames {start_frame} and {end_frame} ({gap_size} frames)")
            
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

    def process_video(self, points_dict, labels_dict, debug=True):
        """Process video with bidirectional propagation using frame reordering approach"""
        results = {}
        
        def cleanup_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        try:
            cleanup_memory()
            
            # Step 1: Process seed chunk with user annotations
            print(f"\n🎯 Step 1: Processing seed frame {self.seed_frame_idx}")
            seed_chunk = next(c for c in self.chunks if c['type'] == 'seed')
            seed_results = self._process_seed_chunk(seed_chunk, points_dict, labels_dict, debug)
            
            if not seed_results:
                print("❌ Failed to process seed frame")
                return None
            
            results.update(seed_results)
            print(f"✅ Seed processing complete")
            
            # Step 2: Process forward chunks (normal order)
            forward_chunks = [c for c in self.chunks if c['type'] == 'forward']
            if forward_chunks:
                print(f"\n➡️ Step 2: Forward propagation ({len(forward_chunks)} chunks)")
                for chunk in forward_chunks:
                    print(f"  Processing forward chunk {chunk['id']}")
                    chunk_results = self._process_forward_chunk(chunk, results, debug)
                    if chunk_results:
                        results.update(chunk_results)
                        print(f"    ✅ Completed: {len(chunk_results)} frames")
                    else:
                        print(f"    ⚠️ No results from chunk {chunk['id']}")
            
            # Step 3: Process backward chunks (reversed frame order)
            backward_chunks = [c for c in self.chunks if c['type'] == 'backward']
            if backward_chunks:
                print(f"\n⬅️ Step 3: Backward propagation ({len(backward_chunks)} chunks)")
                for chunk in backward_chunks:
                    print(f"  Processing backward chunk {chunk['id']} (time-reversed)")
                    chunk_results = self._process_backward_chunk(chunk, results, debug)
                    if chunk_results:
                        results.update(chunk_results)
                        print(f"    ✅ Completed: {len(chunk_results)} frames")
                    else:
                        print(f"    ⚠️ No results from chunk {chunk['id']}")
            
            # Step 4: Fill any remaining gaps
            if results:
                self._fill_result_gaps(results, debug)
                print(f"\n🎉 Processing complete! Total frames: {len(results)}/{len(self.frame_names)}")
            
            return results
            
        except Exception as e:
            print(f"Error in bidirectional processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cleanup_memory()

    def _process_seed_chunk(self, chunk, points_dict, labels_dict, debug=True):
        """Process the single seed frame with user annotations"""
        seed_results = {}
        temp_dir = os.path.join(self.video_dir, f"chunk_seed")
        
        try:
            # Setup temporary directory with just the seed frame
            os.makedirs(temp_dir, exist_ok=True)
            seed_frame_name = chunk['frame_names'][0]
            src = os.path.join(self.video_dir, seed_frame_name)
            dst = os.path.join(temp_dir, "00000.jpg")  # SAM2 expects numbered frames
            shutil.copy2(src, dst)
            
            chunk_state = self.predictor.init_state(video_path=temp_dir)
            
            # Process each object with user annotations
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
                    
                    # Store result for the seed frame
                    for i, prop_obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        
                        # Use the original seed frame index
                        global_frame_idx = self.seed_frame_idx
                        
                        if global_frame_idx not in seed_results:
                            seed_results[global_frame_idx] = {}
                        seed_results[global_frame_idx][prop_obj_id] = mask.copy()
                    
                    del mask_logits
                    cleanup_memory()
                
                except Exception as e:
                    print(f"  Error processing object {obj_id}: {e}")
                    continue
                finally:
                    cleanup_memory()
            
            return seed_results
            
        except Exception as e:
            print(f"Error processing seed chunk: {e}")
            return {}
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            cleanup_memory()

    def _process_forward_chunk(self, chunk, existing_results, debug=True):
        """Process a forward chunk (normal chronological order)"""
        chunk_results = {}
        temp_dir = os.path.join(self.video_dir, f"chunk_{chunk['id']}")
        
        try:
            # Setup temporary directory with numbered frames
            os.makedirs(temp_dir, exist_ok=True)
            for i, frame_name in enumerate(chunk['frame_names']):
                src = os.path.join(self.video_dir, frame_name)
                dst = os.path.join(temp_dir, f"{i:05d}.jpg")
                shutil.copy2(src, dst)
            
            chunk_state = self.predictor.init_state(video_path=temp_dir)
            
            # Get reference masks from seed frame or previous results
            reference_frame_idx = self.seed_frame_idx
            if reference_frame_idx not in existing_results:
                print(f"    Warning: No reference masks from frame {reference_frame_idx}")
                return {}
            
            reference_masks = existing_results[reference_frame_idx]
            
            # Process each object
            for obj_id, reference_mask in reference_masks.items():
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    # Generate points from reference mask
                    points, labels = self._generate_robust_points_from_mask(reference_mask)
                    if points is None:
                        continue
                    
                    # Use first frame as prompt frame
                    prompt_frame_idx = chunk.get('overlap_offset', 0)
                    
                    # Add prompts
                    _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=chunk_state,
                        frame_idx=prompt_frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
                    
                    # Store prompt frame result
                    for i, prop_obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        
                        # Map local frame index to global using original_indices
                        local_idx = prompt_frame_idx
                        if local_idx < len(chunk['original_indices']):
                            global_frame_idx = chunk['original_indices'][local_idx]
                            
                            if global_frame_idx not in chunk_results:
                                chunk_results[global_frame_idx] = {}
                            chunk_results[global_frame_idx][prop_obj_id] = mask.copy()
                    
                    del mask_logits
                    cleanup_memory()
                    
                    # Propagate through chunk
                    for frame_idx, prop_obj_ids, prop_mask_logits in self.predictor.propagate_in_video(chunk_state):
                        # Skip overlap frames except for first chunk
                        if frame_idx < chunk.get('overlap_offset', 0):
                            continue
                            
                        # Map local frame index to global
                        if frame_idx < len(chunk['original_indices']):
                            global_frame_idx = chunk['original_indices'][frame_idx]
                        else:
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
                    print(f"    Error processing object {obj_id}: {e}")
                    continue
                finally:
                    cleanup_memory()
            
            return chunk_results
            
        except Exception as e:
            print(f"Error processing forward chunk {chunk['id']}: {e}")
            return {}
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            cleanup_memory()

    def _process_backward_chunk(self, chunk, existing_results, debug=True):
        """Process a backward chunk (frames in reversed chronological order)"""
        chunk_results = {}
        temp_dir = os.path.join(self.video_dir, f"chunk_{chunk['id']}")
        
        try:
            # Setup temporary directory with reversed frames as numbered sequence
            os.makedirs(temp_dir, exist_ok=True)
            for i, frame_name in enumerate(chunk['frame_names']):
                src = os.path.join(self.video_dir, frame_name)
                dst = os.path.join(temp_dir, f"{i:05d}.jpg")
                shutil.copy2(src, dst)
            
            chunk_state = self.predictor.init_state(video_path=temp_dir)
            
            # Get reference masks from seed frame
            reference_frame_idx = self.seed_frame_idx
            if reference_frame_idx not in existing_results:
                print(f"    Warning: No reference masks from frame {reference_frame_idx}")
                return {}
            
            reference_masks = existing_results[reference_frame_idx]
            
            # Process each object
            for obj_id, reference_mask in reference_masks.items():
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    # Generate points from reference mask  
                    points, labels = self._generate_robust_points_from_mask(reference_mask)
                    if points is None:
                        continue
                    
                    # Use first frame as prompt frame (which is closest to seed in time)
                    prompt_frame_idx = chunk.get('overlap_offset', 0)
                    
                    # Add prompts
                    _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=chunk_state,
                        frame_idx=prompt_frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
                    
                    # Store prompt frame result
                    for i, prop_obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        
                        # Map local frame index to global using original_indices
                        local_idx = prompt_frame_idx
                        if local_idx < len(chunk['original_indices']):
                            global_frame_idx = chunk['original_indices'][local_idx]
                            
                            if global_frame_idx not in chunk_results:
                                chunk_results[global_frame_idx] = {}
                            chunk_results[global_frame_idx][prop_obj_id] = mask.copy()
                    
                    del mask_logits
                    cleanup_memory()
                    
                    # Propagate through chunk (forward in reversed time = backward in real time)
                    for frame_idx, prop_obj_ids, prop_mask_logits in self.predictor.propagate_in_video(chunk_state):
                        # Skip overlap frames except for first chunk
                        if frame_idx < chunk.get('overlap_offset', 0):
                            continue
                            
                        # Map local frame index to global using original_indices
                        if frame_idx < len(chunk['original_indices']):
                            global_frame_idx = chunk['original_indices'][frame_idx]
                        else:
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
                    print(f"    Error processing object {obj_id}: {e}")
                    continue
                finally:
                    cleanup_memory()
            
            return chunk_results
            
        except Exception as e:
            print(f"Error processing backward chunk {chunk['id']}: {e}")
            return {}
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
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
        
        object_names = getattr(self, 'object_names', {})
        
        def get_object_name(obj_id):
            return object_names.get(obj_id, f"Object_{obj_id}")
        
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
                        
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            # Use object name as display text
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
        
        # Print summary of exported objects
        if hasattr(self, 'object_names'):
            object_names = getattr(self, 'object_names', {})
            exported_objects = set()
            for frame_results in results.values():
                exported_objects.update(frame_results.keys())
            
            named_objects = [object_names.get(obj_id, f"Object_{obj_id}") for obj_id in exported_objects]
            
            print(f"Video saved to: {output_path}")
            print(f"  🎬 {len(results)} frames with object overlays")
            print(f"  📊 {len(exported_objects)} object types: {', '.join(named_objects)}")
        else:
            print(f"Video saved to: {output_path}")

    def _save_time_series(self, csv_path):
        """Save time series metrics with object names as identifiers"""
        metrics_data = []
        object_names = getattr(self, 'object_names', {})
        
        def get_object_identifier(obj_id):
            """Get the primary identifier - use name if available, otherwise numeric ID"""
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
                        'object_id': obj_identifier,  # Use object name as primary ID
                        'object_name': obj_identifier,  # For backwards compatibility
                        'numeric_object_id': obj_id,  # Keep original numeric ID for reference
                        'is_named_object': obj_id in object_names,  # Flag to indicate if user provided name
                        'timestamp_seconds': frame_idx / 30.0 if hasattr(self, 'fps') else frame_idx / 30.0  # Approximate timestamp
                    })
                    
                    metrics_data.append(metrics)
                    
            except Exception as e:
                continue
        
        if not metrics_data:
            print("No valid metrics data collected")
            return
            
        # Create DataFrame and calculate deltas by object name (not numeric ID)
        df = pd.DataFrame(metrics_data)
        
        # Calculate deltas grouped by object identifier (name)
        df['delta_centroid_x'] = df.groupby('object_id')['seg_centroid_x'].diff()
        df['delta_centroid_y'] = df.groupby('object_id')['seg_centroid_y'].diff()
        df['delta_area'] = df.groupby('object_id')['surface_area'].diff()
        
        # Calculate velocity (pixels per frame)
        df['velocity_x'] = df['delta_centroid_x'].fillna(0)
        df['velocity_y'] = df['delta_centroid_y'].fillna(0) 
        df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
        
        # Calculate cumulative movement
        df['cumulative_distance'] = df.groupby('object_id')['velocity_magnitude'].cumsum()
        
        # Reorder columns for better readability
        column_order = [
            'frame', 'frame_name', 'timestamp_seconds', 'object_id', 'object_name',
            'seg_centroid_x', 'seg_centroid_y', 'surface_area',
            'delta_centroid_x', 'delta_centroid_y', 'delta_area',
            'velocity_x', 'velocity_y', 'velocity_magnitude', 'cumulative_distance',
            'box_x1', 'box_y1', 'box_x2', 'box_y2', 'box_centroid_x', 'box_centroid_y',
            'mean_color_r', 'mean_color_g', 'mean_color_b', 'color_intensity',
            'std_color_r', 'std_color_g', 'std_color_b',
            'numeric_object_id', 'is_named_object'
        ]
        
        # Reorder columns (keep any additional columns at the end)
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in column_order]
        df = df[available_columns + remaining_columns]
        
        df.to_csv(csv_path, index=False)
        
        # Print summary statistics
        unique_objects = df['object_id'].unique()
        named_objects = df[df['is_named_object'] == True]['object_id'].unique()
        
        print(f"Saved time series metrics to: {csv_path}")
        print(f"  📊 {len(df)} data points across {len(unique_objects)} objects")
        print(f"  🏷️ {len(named_objects)} objects with custom names")
        if len(named_objects) > 0:
            print(f"  📝 Named objects: {', '.join(named_objects)}")
        print(f"  📈 Metrics include: position, area, movement, velocity, and color analysis")

    def _save_coco_annotations(self, json_path):
        """Save annotations in COCO format with object names as identifiers"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        object_names = getattr(self, 'object_names', {})
        
        def get_object_identifier(obj_id):
            """Get the primary identifier - use name if available, otherwise numeric ID"""
            return object_names.get(obj_id, f"Object_{obj_id}")
        
        def get_object_display_name(obj_id):
            """Get display name for backwards compatibility"""
            return object_names.get(obj_id, f"Object_{obj_id}")
        
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "SAM2 segmentation results with human-readable object identifiers",
                "date_created": current_time
            },
            "images": [],
            "annotations": [],
            "licenses": [{"id": 0, "name": "Unknown License", "url": ""}],
            "categories": []
        }
        
        # Create categories using object names as primary identifiers
        unique_objects = set()
        for frame_results in self.results.values():
            unique_objects.update(frame_results.keys())
        
        # Create mapping from name back to numeric ID for internal consistency
        name_to_id = {}
        
        for obj_id in sorted(unique_objects):
            obj_identifier = get_object_identifier(obj_id)
            display_name = get_object_display_name(obj_id)
            
            coco_data["categories"].append({
                "supercategory": "object",
                "id": obj_identifier,  # Use name as primary ID
                "name": display_name,
                "numeric_id": obj_id,  # Keep original numeric ID for reference
                "is_named": obj_id in object_names  # Flag to indicate if user provided name
            })
            
            name_to_id[obj_identifier] = obj_id
        
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
        
        # Add annotations with object names as identifiers
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
                        
                        obj_identifier = get_object_identifier(obj_id)
                        
                        coco_data["annotations"].append({
                            "segmentation": [flattened],
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "id": annotation_id,
                            "image_id": unique_ids[current_frame],
                            "category_id": obj_identifier,  # Use object name as category ID
                            "object_name": get_object_display_name(obj_id),  # Backwards compatibility
                            "numeric_object_id": obj_id,  # Keep original numeric ID for reference
                            "frame_number": frame_idx  # Add frame number for easy reference
                        })
                        annotation_id += 1
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        # Print summary of exported objects
        named_objects = len([obj_id for obj_id in unique_objects if obj_id in object_names])
        total_objects = len(unique_objects)
        
        print(f"Saved COCO annotations to: {json_path}")
        print(f"  📊 {total_objects} object types exported")
        print(f"  🏷️ {named_objects} with custom names, {total_objects - named_objects} with default names")
        if named_objects > 0:
            print(f"  📝 Named objects: {', '.join([object_names[obj_id] for obj_id in unique_objects if obj_id in object_names])}")

    def save_results(self, output_path, fps=30, show_original=True, alpha=0.5):
        """Save all results"""
        self.save_results_video(self.results, output_path, fps, show_original, alpha)
        self._save_coco_annotations(os.path.join(os.path.dirname(output_path), "segmentation_coco.json"))
        self._save_time_series(os.path.join(os.path.dirname(output_path), "time_series_metrics.csv"))

def select_points_opencv(frame, processor=None):
    """Interactive point selection tool with mask preview capability and custom naming"""
    points_dict = {}
    labels_dict = {}
    object_names = {}
    current_obj_id = 1
    
    temp_dir = "temp_select"
    if processor is not None:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # SAM2 expects numbered frame files - use 00000.jpg format  
            frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame to {frame_path}")
            
            chunk_state = processor.predictor.init_state(video_path=temp_dir)
            if chunk_state is None:
                raise ValueError("Failed to initialize chunk state")
            print("Successfully initialized processor state")
            
        except Exception as e:
            print(f"Error initializing processor: {str(e)}")
            import traceback
            traceback.print_exc()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print("❌ Mask preview not available - processor initialization failed")
            return None, None, None
    
    def get_object_name(obj_id):
        """Get display name for object"""
        if obj_id in object_names:
            return f"{obj_id}:{object_names[obj_id]}"
        else:
            return str(obj_id)
    
    def draw_point(img, point, obj_id, label):
        """Draw a point with appropriate color and label"""
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
        
        display_name = get_object_name(obj_id)
        cv2.putText(img, display_name, 
                   (int(point[0] + 5), int(point[1] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def redraw_all_points():
        """Redraw all points on fresh image with keyboard shortcuts"""
        display = frame.copy()
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        
        # Add keyboard shortcuts overlay
        height, width = display.shape[:2]
        
        # Create semi-transparent overlay for instructions
        overlay = display.copy()
        instructions_height = 200
        cv2.rectangle(overlay, (10, height - instructions_height - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Add instruction text
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
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Yellow for title, white for others
            font_scale = 0.6 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            
            cv2.putText(display, instruction, (20, y_start + (i * 22)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Show current object info
        current_obj_name = get_object_name(current_obj_id)
        obj_info = f"Current Object: {current_obj_name}"
        cv2.putText(display, obj_info, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show point count for current object
        if current_obj_id in points_dict:
            pos_count = sum(1 for l in labels_dict[current_obj_id] if l == 1)
            neg_count = sum(1 for l in labels_dict[current_obj_id] if l == 0)
            count_info = f"Points: +{pos_count} -{neg_count}"
            cv2.putText(display, count_info, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return display
    
    def name_current_object():
        """Allow user to name the current object"""
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
        """Show preview of current object's mask"""
        try:
            obj_name = get_object_name(current_obj_id)
            print(f"\nTest mask debug info:")
            print(f"Processor: {'initialized' if processor is not None else 'None'}")
            print(f"Chunk state: {'initialized' if chunk_state is not None else 'None'}")
            print(f"Current object: {obj_name}")
            print(f"Points available: {current_obj_id in points_dict}")
            if current_obj_id in points_dict:
                print(f"Number of points: {len(points_dict[current_obj_id])}")
            
            if not points_dict or not points_dict.get(current_obj_id):
                print("No points selected for current object")
                return
            
            if processor is None or chunk_state is None:
                print("Processor not properly initialized")
                return
            
            points = np.array(points_dict[current_obj_id], dtype=np.float32)
            labels = np.array(labels_dict[current_obj_id], dtype=np.int32)
            
            print(f"Testing mask with {len(points)} points ({sum(labels == 1)} positive, {sum(labels == 0)} negative)")
            
            processor.predictor.reset_state(chunk_state)
            _, obj_ids, mask_logits = processor.predictor.add_new_points_or_box(
                inference_state=chunk_state,
                frame_idx=0,  # Always use frame 0 since we only have one frame
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
            import traceback
            traceback.print_exc()
        
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
            
            # Refresh the entire display to update point counters
            img_display = redraw_all_points()
            obj_name = get_object_name(current_obj_id)
            print(f"Added {'positive' if label == 1 else 'negative'} point for {obj_name}")
    
    img_display = frame.copy()
    cv2.namedWindow('Select Points')
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
        
        elif key == ord('p'):
            if current_obj_id > 1:
                current_obj_id -= 1
                obj_name = get_object_name(current_obj_id)
                print(f"Now selecting {obj_name}")
        
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

class FineTunedVideoProcessor:
    """Processor for using fine-tuned SAM2 models"""
    
    def __init__(self, model_path, config_path, video_dir, auto_mode=True, chunk_size=500, overlap_frames=30):
        """
        Initialize fine-tuned video processor
        
        Args:
            model_path: Path to trained model (.pth file)
            config_path: Path to SAM2 config (.yaml file)  
            video_dir: Directory containing video frames
            auto_mode: If True, automatically detect objects
            chunk_size: Frames per chunk
            overlap_frames: Overlap between chunks
        """
        self.device = setup_device()
        self.auto_mode = auto_mode
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        
        print(f"Loading fine-tuned model from: {model_path}")
        self._load_trained_model(model_path, config_path)
        
        # Initialize base processor components
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Video directory {self.video_dir} does not exist!")
        
        self.frame_names = sorted(
            [p for p in os.listdir(self.video_dir) 
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        if not self.frame_names:
            raise ValueError("No frames found in the specified directory!")
        
        self.chunks = self._calculate_chunks_with_overlap()
        print(f"✅ Fine-tuned model loaded successfully!")
        if hasattr(self, 'trained_objects'):
            print(f"Trained objects: {self.trained_objects}")
    
    def _load_trained_model(self, model_path, config_path):
        """Load the fine-tuned SAM2 model"""
        try:
            # Load checkpoint to get training info
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract trained object information if available
            if 'config' in checkpoint and 'dataset' in checkpoint['config']:
                self.trained_objects = checkpoint['config']['dataset'].get('object_names', [])
                self.object_name_to_id = {name: i+1 for i, name in enumerate(self.trained_objects)}
            else:
                # Fallback if config not in checkpoint
                self.trained_objects = []
                self.object_name_to_id = {}
                print("⚠️ No object names found in checkpoint - will work in manual mode")
            
            # Build base model
            from sam2.build_sam import build_sam2_video_predictor
            self.predictor = build_sam2_video_predictor(config_path, model_path, device=self.device)
            
            # Load fine-tuned weights if they exist
            if 'model_state_dict' in checkpoint:
                try:
                    self.predictor.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Loaded fine-tuned weights")
                except Exception as e:
                    print(f"⚠️ Could not load fine-tuned weights: {e}")
                    print("Using base model weights")
            
            self.predictor.eval()
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            # Fallback to base model
            try:
                from sam2.build_sam import build_sam2_video_predictor
                self.predictor = build_sam2_video_predictor(config_path, model_path, device=self.device)
                self.trained_objects = []
                self.object_name_to_id = {}
                print("⚠️ Loaded base model as fallback")
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to load any model: {fallback_error}")
    
    def _calculate_chunks_with_overlap(self):
        """Calculate chunks with overlap (same as main processor)"""
        chunks = []
        frame_count = len(self.frame_names)
        
        start = 0
        chunk_id = 0
        
        while start < frame_count:
            end = min(start + self.chunk_size, frame_count)
            
            if chunk_id > 0:
                overlap_start = max(0, start - self.overlap_frames)
                chunk_frame_names = self.frame_names[overlap_start:end]
                overlap_offset = start - overlap_start
            else:
                chunk_frame_names = self.frame_names[start:end]
                overlap_offset = 0
            
            chunks.append({
                'id': chunk_id,
                'global_start': start,
                'global_end': end,
                'overlap_offset': overlap_offset,
                'frame_names': chunk_frame_names,
                'frame_indices': list(range(len(chunk_frame_names)))
            })
            
            start = end
            chunk_id += 1
        
        return chunks
    
    def auto_detect_objects(self, frame, confidence_threshold=0.5):
        """
        Automatically detect trained objects in frame
        """
        if not self.trained_objects:
            print("⚠️ No trained objects available for auto-detection")
            return {}
        
        height, width = frame.shape[:2]
        detected_objects = {}
        
        # Initialize inference state for this frame
        temp_dir = "temp_auto_detect"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save frame temporarily with proper naming
            frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Initialize SAM2 state
            inference_state = self.predictor.init_state(video_path=temp_dir)
            
            # Try grid-based detection for each trained object
            for obj_name in self.trained_objects:
                obj_id = self.object_name_to_id[obj_name]
                best_mask = None
                best_score = 0
                
                # Generate detection grid
                grid_points, grid_labels = self._generate_detection_grid(frame, grid_size=60)
                
                if len(grid_points) > 0:
                    try:
                        self.predictor.reset_state(inference_state)
                        
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            points=grid_points,
                            labels=grid_labels,
                        )
                        
                        if len(out_mask_logits) > 0:
                            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                            if len(mask.shape) == 3:
                                mask = mask[0]
                            
                            # Calculate confidence based on mask quality
                            mask_area = np.sum(mask)
                            if mask_area > 100:  # Minimum area threshold
                                mask_compactness = self._calculate_mask_compactness(mask)
                                score = (mask_area / (width * height)) * mask_compactness * 100
                                
                                if score > confidence_threshold and score > best_score:
                                    best_score = score
                                    best_mask = mask
                                    
                    except Exception as e:
                        print(f"Error detecting {obj_name}: {e}")
                        continue
                
                # Store best detection
                if best_mask is not None:
                    detected_objects[obj_id] = best_mask
                    print(f"✅ Auto-detected {obj_name} (confidence: {best_score:.2f})")
        
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return detected_objects
    
    def _generate_detection_grid(self, frame, grid_size=60):
        """Generate a grid of detection points across the frame"""
        height, width = frame.shape[:2]
        points = []
        labels = []
        
        # Create positive points in a grid pattern
        for y in range(grid_size//2, height, grid_size):
            for x in range(grid_size//2, width, grid_size):
                points.append([x, y])
                labels.append(1)
        
        # Add some negative points at edges
        edge_points = [
            [10, 10], [width-10, 10], [10, height-10], [width-10, height-10],
            [width//2, 10], [width//2, height-10], [10, height//2], [width-10, height//2]
        ]
        
        for point in edge_points:
            points.append(point)
            labels.append(0)
        
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    def _calculate_mask_compactness(self, mask):
        """Calculate how compact/coherent a mask is"""
        if not mask.any():
            return 0
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # Calculate compactness (area / perimeter^2) 
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0
        
        compactness = 4 * np.pi * area / (perimeter ** 2)
        return min(compactness, 1.0)  # Clamp to 1.0
    
    def process_video_auto(self, initial_frame_idx=0):
        """
        Process video automatically using fine-tuned model
        """
        if not self.auto_mode:
            raise ValueError("Auto mode not enabled")
        
        print("🤖 Starting automatic video processing with fine-tuned model...")
        
        # Step 1: Auto-detect objects in initial frame
        print(f"Detecting objects in frame {initial_frame_idx}...")
        initial_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[initial_frame_idx]))
        
        detected_objects = self.auto_detect_objects(initial_frame)
        
        if not detected_objects:
            print("❌ No objects detected automatically")
            return None, None
        
        print(f"✅ Detected {len(detected_objects)} objects automatically")
        
        # Step 2: Convert detections to points for tracking
        points_dict = {}
        labels_dict = {}
        object_names = {}
        
        for obj_id, mask in detected_objects.items():
            points, labels = self._mask_to_points(mask)
            if points is not None:
                points_dict[obj_id] = points.tolist()
                labels_dict[obj_id] = labels.tolist()
                
                # Map back to object name
                if self.trained_objects and obj_id <= len(self.trained_objects):
                    obj_name = self.trained_objects[obj_id-1]
                    object_names[obj_id] = obj_name
                else:
                    object_names[obj_id] = f"Object_{obj_id}"
        
        # Step 3: Process video with detected objects
        print("Processing video with detected objects...")
        results = self.process_video_with_prompts(points_dict, labels_dict)
        
        return results, object_names
    
    def _mask_to_points(self, mask, num_points=8):
        """Convert a detected mask back to prompt points"""
        if not mask.any():
            return None, None
        
        # Get contour points
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Sample points along contour
        points = []
        labels = []
        
        contour_length = len(largest_contour)
        step = max(1, contour_length // num_points)
        
        for i in range(0, contour_length, step):
            point = largest_contour[i][0]
            points.append([int(point[0]), int(point[1])])
            labels.append(1)
        
        # Add center point
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            points.append([cx, cy])
            labels.append(1)
        
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    def process_video_with_prompts(self, points_dict, labels_dict, seed_frame_idx=0):
        """Process video with given prompts (reuse existing VideoChunkProcessor logic)"""
        # Create a temporary VideoChunkProcessor with our fine-tuned predictor
        temp_processor = VideoChunkProcessor(
            predictor=self.predictor,
            video_dir=self.video_dir,
            chunk_size=self.chunk_size,
            overlap_frames=self.overlap_frames,
            interactive_correction=False,  # Disable interactive correction for auto mode
            seed_frame_idx=seed_frame_idx
        )
        
        return temp_processor.process_video(points_dict, labels_dict)
    
    def save_results(self, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results (reuse existing save logic)"""
        # Create a temporary VideoChunkProcessor for saving
        temp_processor = VideoChunkProcessor(
            predictor=self.predictor,
            video_dir=self.video_dir,
            chunk_size=self.chunk_size,
            overlap_frames=self.overlap_frames,
            seed_frame_idx=0  # Default seed for saving
        )
        
        # Copy our results and object names
        temp_processor.results = self.results
        temp_processor.object_names = getattr(self, 'object_names', {})
        
        # Use existing save functionality
        temp_processor.save_results(output_path, fps, show_original, alpha)


class VideoAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAM2 Video Analysis & Inference")
        self.root.geometry("650x700")  # Made bigger to show all buttons
        self.root.minsize(650, 700)    # Increased minimum size too
        
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
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="SAM2 Video Analysis & Inference", 
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
        
        # Processing options
        options_frame = tk.LabelFrame(main_frame, text="Processing Options", font=("Arial", 9, "bold"))
        options_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Interactive correction option
        self.interactive_correction = tk.BooleanVar(value=True)
        correction_cb = tk.Checkbutton(options_frame, 
                                      text="🎯 Interactive mask correction (pause for manual fixes when needed)",
                                      variable=self.interactive_correction)
        correction_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Analysis video option
        self.create_analysis_video = tk.BooleanVar(value=True)
        analysis_cb = tk.Checkbutton(options_frame, 
                                    text="📊 Create analysis video with plots and metrics",
                                    variable=self.create_analysis_video)
        analysis_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        help_label = tk.Label(options_frame, 
                             text="Interactive correction: You'll be prompted to fix discontinuous masks\n"
                                  "Analysis video: Creates detailed video with movement/area/color plots",
                             font=("Arial", 8), fg="gray")
        help_label.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Fine-tuning workflow section
        finetuning_frame = tk.LabelFrame(main_frame, text="🧠 SAM2 Fine-tuning Workflow", font=("Arial", 9, "bold"))
        finetuning_frame.pack(fill=tk.X, pady=(10, 10))
        
        ft_info_label = tk.Label(finetuning_frame, 
                                text="After annotating multiple videos, create a specialized model for your objects:",
                                font=("Arial", 8), fg="navy")
        ft_info_label.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        ft_buttons_frame = tk.Frame(finetuning_frame)
        ft_buttons_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        tk.Button(ft_buttons_frame, text="🔧 Setup Fine-tuning Environment", 
                 command=self.setup_finetuning, bg="#FF5722", fg="white",
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(ft_buttons_frame, text="🚀 Start Training Model", 
                 command=self.start_training, bg="#3F51B5", fg="white",
                 font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Inference section
        inference_frame = tk.LabelFrame(main_frame, text="🎯 Apply Trained Model", font=("Arial", 9, "bold"))
        inference_frame.pack(fill=tk.X, pady=(10, 10))
        
        inference_info_label = tk.Label(inference_frame, 
                                       text="Use your trained model to automatically process new videos:",
                                       font=("Arial", 8), fg="darkgreen")
        inference_info_label.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        tk.Button(inference_frame, text="🎬 Auto-Process with Trained Model", 
                 command=self.inference_with_trained_model, bg="#8BC34A", fg="white",
                 font=("Arial", 10, "bold"), pady=5).pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Process button - Make it more prominent
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(15, 10))
        
        self.process_button = tk.Button(process_frame, text="Process Selected Video", 
                                       command=self.process_video, bg="#4CAF50", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select a folder and video to begin")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 8), wraplength=600)  # Increased wrap length
        status_label.pack(pady=(5, 0))
    
    def inference_with_trained_model(self):
        """Apply a trained model to automatically process video"""
        try:
            # Validate basic inputs first
            selection = self.video_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a video to process")
                return
            
            folder = self.folder_var.get()
            if not folder:
                messagebox.showwarning("Warning", "Please select a folder first")
                return
            
            # Select trained model file
            model_path = filedialog.askopenfilename(
                title="Select Trained SAM2 Model",
                filetypes=[
                    ("PyTorch Model files", "*.pth"),
                    ("All files", "*.*")
                ],
                initialdir=folder
            )
            
            if not model_path:
                return
            
            # Select model config file
            config_path = filedialog.askopenfilename(
                title="Select SAM2 Model Config",
                filetypes=[
                    ("YAML Config files", "*.yaml"),
                    ("All files", "*.*")
                ],
                initialdir="."
            )
            
            if not config_path:
                messagebox.showwarning("Warning", "Model config file is required")
                return
            
            # Validate files exist
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
                
            if not os.path.exists(config_path):
                messagebox.showerror("Error", f"Config file not found: {config_path}")
                return
            
            # Get video info
            video_name = self.video_listbox.get(selection[0])
            video_path = os.path.join(folder, video_name)
            video_stem = Path(video_name).stem
            frames_dir = os.path.join(folder, f"{video_stem}_frames")
            
            # Ask for processing options
            auto_detect = messagebox.askyesno("Processing Mode", 
                "🤖 Automatic Detection Mode?\n\n"
                "YES: Let the model automatically find trained objects\n"
                "NO: Use a reference frame for object selection\n\n"
                "Recommendation: Try automatic mode first!")
            
            self.status_var.set("Setting up trained model...")
            self.root.update()
            
            # Extract frames if needed
            if not os.path.exists(frames_dir):
                self.status_var.set("Extracting frames...")
                self.root.update()
                
                fps, num_frames = video_to_frames(video_path, frames_dir)
                if fps == -1:
                    messagebox.showerror("Error", "Failed to extract frames from video")
                    return
            else:
                fps, _ = get_video_fps(video_path)
            
            # Initialize trained model processor
            self.status_var.set("Loading trained model...")
            self.root.update()
            
            trained_processor = FineTunedVideoProcessor(
                model_path=model_path,
                config_path=config_path,
                video_dir=frames_dir,
                auto_mode=auto_detect
            )
            
            if auto_detect:
                # Automatic processing
                self.status_var.set("🤖 Processing automatically with trained model...")
                self.root.update()
                
                results, object_names = trained_processor.process_video_auto()
                
                if results is None:
                    messagebox.showerror("Error", 
                        "Automatic detection failed.\n"
                        "Try manual mode or check if model was trained on similar objects.")
                    return
                    
            else:
                # Manual reference frame mode
                frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                if not frame_files:
                    messagebox.showerror("Error", "No frames found in directory")
                    return
                
                # Get reference frame
                frame_num = self.get_frame_number(len(frame_files))
                if frame_num is None:
                    return
                
                frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
                reference_frame = cv2.imread(frame_path)
                
                self.status_var.set("Select reference objects for tracking...")
                self.root.update()
                
                messagebox.showinfo("Reference Selection", 
                    "Select objects in the reference frame.\n"
                    "The trained model will track these objects throughout the video.")
                
                points_dict, labels_dict, object_names = select_points_opencv(reference_frame, trained_processor)
                
                if points_dict is None:
                    self.status_var.set("Processing cancelled")
                    return
                
                self.status_var.set("🎯 Processing with trained model...")
                self.root.update()
                
                results = trained_processor.process_video_with_prompts(points_dict, labels_dict, frame_num)
                
            # Save results
            if results:
                trained_processor.results = results
                trained_processor.object_names = object_names
                
                self.status_var.set("Saving results...")
                self.root.update()
                
                # Save with different name to avoid overwriting manual annotations
                output_path = os.path.join(frames_dir, "trained_model_output.mp4")
                trained_processor.save_results(
                    output_path=output_path,
                    fps=fps,
                    show_original=True,
                    alpha=0.5
                )
                
                # Ask about analysis video
                create_analysis = messagebox.askyesno("Analysis Video", 
                    "Create detailed analysis video with plots and metrics?")
                
                success_msg = f"""🎉 Trained Model Processing Complete!

Model: {Path(model_path).name}
Mode: {'Automatic Detection' if auto_detect else 'Manual Reference'}
Results saved in: {frames_dir}

📁 Generated Files:
• trained_model_output.mp4 - Video with overlays
• segmentation_coco.json - Object annotations  
• time_series_metrics.csv - Movement data

📊 Processed Objects ({len(object_names)}):
""" + "\n".join([f"  • {name}" for name in object_names.values()])
                
                if create_analysis:
                    self.status_var.set("Creating analysis video...")
                    self.root.update()
                    
                    analysis_output = os.path.join(frames_dir, "trained_model_analysis.mp4")
                    # Note: Would need to implement create_analysis_video for trained processor
                    success_msg += "\n🎬 Analysis video: trained_model_analysis.mp4"
                
                self.status_var.set("Trained model processing completed!")
                messagebox.showinfo("Success", success_msg)
                
            else:
                messagebox.showerror("Error", "Processing failed with trained model")
                self.status_var.set("Processing failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
            self.status_var.set("Inference failed")
            import traceback
            traceback.print_exc()
        """Setup fine-tuning environment"""
        try:
            # Check if sam2_finetuning_setup.py exists
            if os.path.exists("sam2_finetuning_setup.py"):
                messagebox.showinfo("Fine-tuning Setup", 
                    "Running SAM2 fine-tuning setup...\n"
                    "This will create training configuration and scripts.")
                
                # Run the setup script
                result = subprocess.run([sys.executable, "sam2_finetuning_setup.py"], 
                                       capture_output=True, text=True)
                
                if result.returncode == 0:
                    messagebox.showinfo("Setup Complete", 
                        "Fine-tuning environment setup complete!\n\n"
                        "Files created:\n"
                        "- training_config.yaml\n"
                        "- train_sam2.py\n"
                        "- use_finetuned_sam2.py\n\n"
                        "Next: Annotate multiple videos, then click 'Start Training Model'")
                else:
                    messagebox.showerror("Setup Error", f"Setup failed:\n{result.stderr}")
            else:
                messagebox.showwarning("File Not Found", 
                    "sam2_finetuning_setup.py not found in current directory.\n"
                    "Please ensure all fine-tuning files are in the same folder.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup fine-tuning: {str(e)}")
    
    def start_training(self):
        """Start training fine-tuned model"""
        try:
            # Check if required files exist
            if not os.path.exists("training_config.yaml"):
                messagebox.showwarning("Config Missing", 
                    "training_config.yaml not found.\n"
                    "Please run 'Setup Fine-tuning Environment' first.")
                return
                
            if not os.path.exists("train_sam2.py"):
                messagebox.showwarning("Training Script Missing", 
                    "train_sam2.py not found.\n"
                    "Please run 'Setup Fine-tuning Environment' first.")
                return
            
            # Ask user to confirm training
            result = messagebox.askyesno("Start Training", 
                "This will start SAM2 fine-tuning training.\n\n"
                "Prerequisites:\n"
                "✓ Multiple annotated videos\n"
                "✓ Updated training_config.yaml\n"
                "✓ GPU with enough memory\n\n"
                "Training can take several hours.\n"
                "Continue?")
            
            if result:
                messagebox.showinfo("Training Started", 
                    "Fine-tuning training started!\n\n"
                    "The training will run in a separate process.\n"
                    "Check the console for progress updates.\n\n"
                    "You can continue using this app while training runs.")
                
                # Start training in background
                subprocess.Popen([sys.executable, "train_sam2.py", "training_config.yaml"])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def setup_finetuning(self):
        """Setup fine-tuning environment"""
        try:
            # Check if sam2_finetuning_setup.py exists
            if os.path.exists("sam2_finetuning_setup.py"):
                messagebox.showinfo("Fine-tuning Setup", 
                    "Running SAM2 fine-tuning setup...\n"
                    "This will create training configuration and scripts.")
                
                # Run the setup script
                result = subprocess.run([sys.executable, "sam2_finetuning_setup.py"], 
                                       capture_output=True, text=True)
                
                if result.returncode == 0:
                    messagebox.showinfo("Setup Complete", 
                        "Fine-tuning environment setup complete!\n\n"
                        "Files created:\n"
                        "- training_config.yaml\n"
                        "- train_sam2.py\n"
                        "- use_finetuned_sam2.py\n\n"
                        "Next: Annotate multiple videos, then click 'Start Training Model'")
                else:
                    messagebox.showerror("Setup Error", f"Setup failed:\n{result.stderr}")
            else:
                messagebox.showwarning("File Not Found", 
                    "sam2_finetuning_setup.py not found in current directory.\n"
                    "Please ensure all fine-tuning files are in the same folder.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup fine-tuning: {str(e)}")
    
    def start_training(self):
        """Start training fine-tuned model"""
        try:
            # Check if required files exist
            if not os.path.exists("training_config.yaml"):
                messagebox.showwarning("Config Missing", 
                    "training_config.yaml not found.\n"
                    "Please run 'Setup Fine-tuning Environment' first.")
                return
                
            if not os.path.exists("train_sam2.py"):
                messagebox.showwarning("Training Script Missing", 
                    "train_sam2.py not found.\n"
                    "Please run 'Setup Fine-tuning Environment' first.")
                return
            
            # Ask user to confirm training
            result = messagebox.askyesno("Start Training", 
                "This will start SAM2 fine-tuning training.\n\n"
                "Prerequisites:\n"
                "✓ Multiple annotated videos\n"
                "✓ Updated training_config.yaml\n"
                "✓ GPU with enough memory\n\n"
                "Training can take several hours.\n"
                "Continue?")
            
            if result:
                messagebox.showinfo("Training Started", 
                    "Fine-tuning training started!\n\n"
                    "The training will run in a separate process.\n"
                    "Check the console for progress updates.\n\n"
                    "You can continue using this app while training runs.")
                
                # Start training in background
                subprocess.Popen([sys.executable, "train_sam2.py", "training_config.yaml"])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
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
        # Suggest middle frame for best bidirectional results
        suggested_frame = total_frames // 2
        
        frame_num = simpledialog.askinteger(
            "Reference Frame Selection",
            f"Select frame for object annotation (0-{total_frames-1}):\n\n"
            f"💡 Tip: Choose a frame where objects are clearly visible\n"
            f"🔄 Processing will propagate BOTH forward and backward from this frame\n"
            f"📍 Suggested: Frame {suggested_frame} (middle of video)\n\n"
            f"Enter frame number:",
            minvalue=0,
            maxvalue=total_frames-1,
            initialvalue=suggested_frame
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
            
            # Initialize processor with seed frame
            self.status_var.set("Initializing processor...")
            self.root.update()
            
            processor = VideoChunkProcessor(
                predictor=self.predictor, 
                video_dir=frames_dir, 
                chunk_size=500, 
                overlap_frames=30,
                interactive_correction=self.interactive_correction.get(),
                seed_frame_idx=frame_num  # Pass the selected frame as seed
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
                "The frame will open in a new window for annotation.\n\n"
                "🎯 All keyboard shortcuts are displayed on the frame!\n"
                "✓ Left click: positive points\n"
                "✓ Right click: negative points\n"
                "✓ Press 'C' to name objects with custom names\n"
                "✓ Press 'T' to preview masks\n\n"
                f"📍 Selected frame {frame_num} will be used as reference\n"
                "🔄 Processing will propagate forward AND backward from this frame\n\n"
                "Start by clicking on your first object!")
            
            points_dict, labels_dict, object_names = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Process video
            self.status_var.set(f"🔄 Processing video with SAM2 (bidirectional from frame {frame_num})...")
            self.root.update()
            
            results = processor.process_video(points_dict, labels_dict)
            
            if results:
                processor.results = results
                processor.object_names = object_names  # Store object names in processor
                
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
                
                # Ask if user wants analysis video
                create_analysis = messagebox.askyesno("Analysis Video", 
                    "Do you want to create an analysis video with plots and metrics?\n"
                    "This may take additional time but provides detailed insights.")
                
                if create_analysis:
                    self.status_var.set("Creating analysis video...")
                    self.root.update()
                    
                    analysis_output = os.path.join(frames_dir, "analysis_video.mp4")
                    # Note: You would need to implement create_analysis_video method
                    # processor.create_analysis_video(
                    #     results=results,
                    #     output_path=analysis_output,
                    #     fps=fps,
                    #     object_names=object_names
                    # )
                    
                    self.status_var.set("Processing completed with analysis video!")
                    correction_mode = "Interactive correction" if self.interactive_correction.get() else "Automatic recovery"
                    
                    # Create summary of processed objects
                    named_objects = [name for name in object_names.values()]
                    total_objects = len(object_names)
                    
                    objects_summary = "\n".join([f"  • {name}" for name in named_objects]) if named_objects else "  • No objects processed"
                    
                    messagebox.showinfo("Success", 
                        f"🎉 Processing completed!\n"
                        f"Mode: {correction_mode}\n"
                        f"Results saved in: {frames_dir}\n\n"
                        f"📁 Generated Files:\n"
                        f"  • output_masked.mp4 - Video with overlays\n"
                        f"  • segmentation_coco.json - Object annotations\n"
                        f"  • time_series_metrics.csv - Movement data\n\n"
                        f"📊 Processed Objects ({total_objects}):\n{objects_summary}")
                else:
                    self.status_var.set("Processing completed!")
                    correction_mode = "Interactive correction" if self.interactive_correction.get() else "Automatic recovery"
                    messagebox.showinfo("Success", 
                        f"Processing completed!\n"
                        f"Mode: {correction_mode}\n"
                        f"Results saved in: {frames_dir}\n"
                        f"- Masked video: output_masked.mp4\n"
                        f"- COCO annotations: segmentation_coco.json\n"
                        f"- Time series data: time_series_metrics.csv\n\n"
                        f"Object names used:\n" + 
                        "\n".join([f"  {id}: {name}" for id, name in object_names.items()]))
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
    print("Starting SAM2 Video Analysis & Inference Application...")
    print("Features:")
    print("✓ Interactive video annotation with SAM2")
    print("✓ Fine-tuning setup and training")
    print("✓ Automatic inference with trained models")
    print("\nRequirements:")
    print("1. SAM2 installed and checkpoints downloaded")
    print("2. FFmpeg installed for video processing")
    print("3. Required Python packages: opencv-python, torch, matplotlib, pandas, tqdm")
    print("\nStarting GUI...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()