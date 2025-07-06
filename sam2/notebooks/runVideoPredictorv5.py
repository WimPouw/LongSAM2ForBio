#!/usr/bin/env python3
"""
Complete SAM2 Video Analysis Script - Enhanced with Target Overlap Detection
Addresses all major issues + adds target overlap tracking and ELAN export
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
import yaml
import time  # Added for ELAN timestamps
from datetime import datetime
import gc
from tqdm import tqdm
import pandas as pd

# Training-related imports (loaded globally for class definitions)
try:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from torchvision.models.segmentation import deeplabv3_resnet50
    PYTORCH_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some PyTorch components not available: {e}")
    PYTORCH_TRAINING_AVAILABLE = False
    # Create dummy Dataset class to avoid import errors
    class Dataset:
        pass

# Additional imports for training (loaded dynamically to avoid dependencies)
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

class TargetOverlapTracker:
    """Tracks overlaps between target objects and other objects"""
    
    def __init__(self, overlap_threshold=0.1):
        """
        Initialize overlap tracker
        
        Args:
            overlap_threshold: Minimum overlap percentage (0.0-1.0) to register as overlap
        """
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}  # {target_id: [events]}
        self.target_objects = {}  # {obj_id: target_name}
        
    def register_target(self, obj_id, obj_name):
        """Register an object as a target if it has 'target' in the name"""
        if 'target' in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"🎯 Registered target: {obj_name} (ID: {obj_id})")
            return True
        return False
    
    def calculate_overlap(self, mask1, mask2):
        """Calculate overlap percentage between two masks"""
        if mask1.shape != mask2.shape:
            return 0.0
        
        # Ensure masks are boolean
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1_bool, mask2_bool)
        area_intersection = np.sum(intersection)
        
        # Calculate overlap as percentage of smaller object
        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # Use the smaller area as denominator for overlap percentage
        smaller_area = min(area1, area2)
        overlap_percentage = area_intersection / smaller_area
        
        return overlap_percentage
    
    def track_frame_overlaps(self, frame_idx, frame_results, object_names):
        """Track overlaps for a single frame"""
        if not self.target_objects or not frame_results:
            return
        
        # Check each target
        for target_id in self.target_objects:
            if target_id not in frame_results:
                continue
            
            target_mask = frame_results[target_id]
            target_name = self.target_objects[target_id]
            overlapping_objects = []
            
            # Check overlap with all other objects
            for obj_id, mask in frame_results.items():
                if obj_id == target_id:  # Skip self
                    continue
                
                overlap_pct = self.calculate_overlap(target_mask, mask)
                
                if overlap_pct >= self.overlap_threshold:
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    overlapping_objects.append({
                        'object_id': obj_id,
                        'object_name': obj_name,
                        'overlap_percentage': overlap_pct
                    })
            
            # Record overlap event
            if overlapping_objects:
                # Create or update current overlap event
                self._update_overlap_event(target_id, frame_idx, overlapping_objects)
    
    def _update_overlap_event(self, target_id, frame_idx, overlapping_objects):
        """Update or create overlap event for target"""
        events = self.overlap_events[target_id]
        
        # Get names of overlapping objects
        current_overlap_names = set(obj['object_name'] for obj in overlapping_objects)
        
        # Check if this continues the last event
        if events and not events[-1].get('end_frame'):
            last_event = events[-1]
            last_overlap_names = set(last_event['overlapping_objects'])
            
            # If same objects are overlapping, extend the event
            if current_overlap_names == last_overlap_names:
                last_event['end_frame'] = frame_idx
                last_event['duration_frames'] = frame_idx - last_event['start_frame'] + 1
                return
            else:
                # Different objects, close previous event
                last_event['end_frame'] = frame_idx - 1
                last_event['duration_frames'] = last_event['end_frame'] - last_event['start_frame'] + 1
        
        # Start new overlap event
        new_event = {
            'start_frame': frame_idx,
            'end_frame': None,  # Will be set when event ends
            'duration_frames': 1,
            'overlapping_objects': list(current_overlap_names),
            'overlap_details': overlapping_objects
        }
        
        events.append(new_event)
    
    def finalize_tracking(self, last_frame_idx):
        """Finalize any open overlap events"""
        for target_id, events in self.overlap_events.items():
            if events and not events[-1].get('end_frame'):
                events[-1]['end_frame'] = last_frame_idx
                events[-1]['duration_frames'] = last_frame_idx - events[-1]['start_frame'] + 1
    
    def get_overlap_summary(self):
        """Get summary of all overlap events"""
        summary = {}
        
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            summary[target_name] = {
                'total_events': len(events),
                'events': events,
                'total_overlap_frames': sum(event['duration_frames'] for event in events)
            }
        
        return summary
    
    def has_targets(self):
        """Check if any targets are registered"""
        return bool(self.target_objects)

def create_elan_file_with_targets(
    video_path: str,
    overlap_tracker: TargetOverlapTracker,
    output_path: str,
    fps: float,
    object_names: dict = None
) -> None:
    """
    Create ELAN file from target overlap events
    
    Args:
        video_path: Path to the source video file (masked video)
        overlap_tracker: TargetOverlapTracker with recorded events
        output_path: Path to save the ELAN file
        fps: Video frame rate
        object_names: Dictionary mapping object IDs to names
    """
    if not overlap_tracker.has_targets():
        print("No targets found - skipping ELAN export")
        return
    
    print(f"Creating ELAN file: {output_path}")
    
    # Get overlap summary
    summary = overlap_tracker.get_overlap_summary()
    
    # Create the basic ELAN file structure
    header = f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="SAM2_VideoAnalysis" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
'''

    # Collect all time points and create time slots
    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}  # Store references for annotations
    
    all_time_points = set()
    
    # Collect all start and end times from overlap events
    for target_name, target_data in summary.items():
        for event in target_data['events']:
            start_time = event['start_frame'] / fps
            end_time = event['end_frame'] / fps
            all_time_points.add(start_time)
            all_time_points.add(end_time)
    
    # Create time slots for all unique time points
    for time_point in sorted(all_time_points):
        time_ms = int(time_point * 1000)
        time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
        time_slot_refs[time_ms] = f"ts{time_slot_id}"
        time_slot_id += 1

    # Add time slots to header
    header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

    # Create tiers for each target
    tier_content = ""
    annotation_id = 1
    
    for target_name, target_data in summary.items():
        tier_id = target_name.upper().replace(' ', '_')
        tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}">\n'
        
        # Add annotations for each overlap event
        for event in target_data['events']:
            start_time = event['start_frame'] / fps
            end_time = event['end_frame'] / fps
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            start_slot = time_slot_refs[start_ms]
            end_slot = time_slot_refs[end_ms]
            
            # Create annotation value with overlapping objects
            overlapping_objects_str = ", ".join(event['overlapping_objects'])
            annotation_value = f"Overlap: {overlapping_objects_str}"
            
            annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''
            
            tier_content += annotation + '\n'
            annotation_id += 1
        
        tier_content += '    </TIER>\n'

    # Add linguistic type definitions
    footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>'''

    # Write the complete ELAN file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + tier_content + footer)
    
    print(f"✅ ELAN file created: {output_path}")
    
    # Print summary
    print("\n📊 Target Overlap Summary:")
    for target_name, target_data in summary.items():
        print(f"  🎯 {target_name}:")
        print(f"    • Total events: {target_data['total_events']}")
        print(f"    • Total overlap frames: {target_data['total_overlap_frames']}")
        print(f"    • Total overlap time: {target_data['total_overlap_frames']/fps:.2f}s")

class VideoChunkProcessor:
    def __init__(self, predictor, video_dir, chunk_size=500, overlap_frames=20, 
                 interactive_correction=True, seed_frame_idx=0, overlap_threshold=0.1):
        self.predictor = predictor
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.interactive_correction = interactive_correction
        self.seed_frame_idx = seed_frame_idx
        
        # Add overlap tracking
        self.overlap_tracker = TargetOverlapTracker(overlap_threshold)
        self.overlap_threshold = overlap_threshold
        
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
        print(f"🎯 Overlap tracking enabled (threshold: {overlap_threshold*100:.1f}%)")
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
        """Process video with improved bidirectional propagation and overlap tracking"""
        results = {}
        
        try:
            cleanup_memory()
            
            # Store object names for overlap tracking
            self.object_names = getattr(self, 'object_names', {})
            
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
            
            # Step 5: Target overlap tracking - ENHANCED
            if hasattr(self, 'overlap_tracker') and self.object_names:
                # Register targets FIRST
                targets_found = False
                for obj_id, obj_name in self.object_names.items():
                    if self.overlap_tracker.register_target(obj_id, obj_name):
                        targets_found = True
                
                if targets_found:
                    print(f"\n🎯 Tracking target overlaps (threshold: {self.overlap_threshold*100:.1f}%)...")
                    
                    # Track overlaps frame by frame
                    overlap_count = 0
                    for frame_idx in tqdm(sorted(results.keys()), desc="Tracking overlaps"):
                        if frame_idx in results:
                            self.overlap_tracker.track_frame_overlaps(frame_idx, results[frame_idx], self.object_names)
                            
                            # Count overlaps for debugging
                            for target_id in self.overlap_tracker.target_objects:
                                if target_id in results[frame_idx]:
                                    target_mask = results[frame_idx][target_id]
                                    for obj_id, mask in results[frame_idx].items():
                                        if obj_id != target_id:
                                            overlap_pct = self.overlap_tracker.calculate_overlap(target_mask, mask)
                                            if overlap_pct >= self.overlap_tracker.overlap_threshold:
                                                overlap_count += 1
                                                if debug and overlap_count <= 5:  # Only show first few
                                                    obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                                                    target_name = self.overlap_tracker.target_objects[target_id]
                                                    print(f"    Frame {frame_idx}: {target_name} overlaps with {obj_name} ({overlap_pct*100:.1f}%)")
                    
                    # Finalize tracking
                    last_frame = max(results.keys()) if results else 0
                    self.overlap_tracker.finalize_tracking(last_frame)
                    
                    print(f"✅ Target overlap tracking completed - {overlap_count} total overlaps detected")
                    
                    # Print summary
                    summary = self.overlap_tracker.get_overlap_summary()
                    print("\n📊 Overlap Summary:")
                    for target_name, data in summary.items():
                        print(f"  🎯 {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames")
                        if data['total_events'] > 0:
                            print(f"      Average event duration: {data['total_overlap_frames']/data['total_events']:.1f} frames")
                else:
                    print("ℹ️  No target objects found (use 'target_1', 'target_2', etc. for overlap tracking)")
            
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
        """
        Create enhanced analysis video with:
        - Masked overlay with connection lines to time series
        - Target overlap highlighting
        - Movement and area plots only (simplified)
        
        Args:
            results: Dictionary of results from process_video
            output_path: Where to save the analysis video
            fps: Frames per second for output video
            alpha: Opacity of mask overlay (0 to 1)
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        if not results:
            print("No results to analyze!")
            return
        
        # Get color map and object info
        cmap = plt.get_cmap("tab10")
        object_names = getattr(self, 'object_names', {})
        overlap_tracker = getattr(self, 'overlap_tracker', None)
        
        # Identify targets
        target_objects = {}
        if overlap_tracker and overlap_tracker.has_targets():
            target_objects = overlap_tracker.target_objects
            print(f"🎯 Creating analysis video with {len(target_objects)} targets: {list(target_objects.values())}")
        
        # Pre-calculate all frame overlaps for the entire video
        print("Pre-calculating overlap data for analysis video...")
        frame_overlaps = {}  # Store overlap info per frame
        
        if overlap_tracker and target_objects:
            for frame_idx in sorted(results.keys()):
                frame_overlaps[frame_idx] = {}
                
                for target_id in target_objects:
                    if target_id in results[frame_idx]:
                        target_mask = results[frame_idx][target_id]
                        overlapping = []
                        
                        for obj_id, mask in results[frame_idx].items():
                            if obj_id != target_id:
                                overlap_pct = overlap_tracker.calculate_overlap(target_mask, mask)
                                if overlap_pct >= overlap_tracker.overlap_threshold:
                                    overlapping.append(obj_id)
                        
                        if overlapping:
                            frame_overlaps[frame_idx][target_id] = overlapping
            
            # Count total overlaps for debugging
            total_overlap_frames = len([f for f in frame_overlaps.values() if any(f.values())])
            print(f"Found overlaps in {total_overlap_frames} frames")
        
        # Collect time series data
        print("Collecting time series data...")
        time_series_data = {}
        max_frame_idx = max(results.keys())
        
        for obj_id in set(obj_id for frame in results.values() for obj_id in frame.keys()):
            time_series_data[obj_id] = {
                'frames': [],
                'centroids': [],
                'areas': [],
                'plot_color': cmap(obj_id % 10)[:3],  # Store plot color to match mask
                'is_target': obj_id in target_objects
            }
        
        # Calculate metrics for all frames
        print("Calculating metrics...")
        for frame_idx in sorted(results.keys()):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            
            # Calculate standard metrics
            for obj_id, mask in results[frame_idx].items():
                box = self._compute_box_from_mask(mask)
                metrics = self._compute_metrics(mask, box, frame)
                if metrics is None:
                    continue
                    
                data = time_series_data[obj_id]
                data['frames'].append(frame_idx)
                data['centroids'].append((metrics['seg_centroid_x'], metrics['seg_centroid_y']))
                data['areas'].append(metrics['surface_area'])
        
        # Calculate derived metrics
        print("Calculating derived metrics...")
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
            
            # Calculate moving averages
            data['area_ma'] = np.convolve(data['areas'], 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
            data['movement_ma'] = np.convolve(data['movement'],
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
        
        # Total output dimensions
        out_width = width + (2 * side_plot_width)
        out_height = height
        
        # Video position in output frame
        video_x = side_plot_width
        video_y = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        print("\nCreating enhanced analysis video with target overlap detection...")
        overlap_frame_count = 0
        
        for frame_idx in tqdm(range(len(self.frame_names)), desc="Creating analysis video"):
            # Create output canvas
            output_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            # Create masked overlay
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            overlay = frame.copy()
            
            # Store mask centroids for connection lines
            centroids = {}
            
            # Check if this frame has overlaps
            has_overlaps = frame_idx in frame_overlaps and any(frame_overlaps[frame_idx].values())
            if has_overlaps:
                overlap_frame_count += 1
            
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
                    
                    # Determine if this object is overlapping with targets - FIXED
                    is_overlapping = False
                    overlap_targets = []
                    
                    if frame_idx in frame_overlaps:
                        # Check if this object is a target that's overlapping
                        if obj_id in frame_overlaps[frame_idx]:
                            is_overlapping = True
                            overlap_targets.append(obj_id)
                        
                        # Check if this object is overlapping with any target
                        for target_id, overlapping_objects in frame_overlaps[frame_idx].items():
                            if obj_id in overlapping_objects:
                                is_overlapping = True
                                overlap_targets.append(target_id)
                    
                    # Choose color - highlight overlapping objects
                    base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                    if is_overlapping:
                        # Add red tint for overlapping objects
                        color = np.minimum(base_color + [80, 0, 0], 255)  # More pronounced red
                        border_color = (0, 0, 255)  # Red border
                    else:
                        color = base_color
                        border_color = None
                    
                    # Apply mask color
                    color_mask = np.zeros_like(overlay)
                    for c in range(3):
                        color_mask[:, :, c][mask] = color[c]
                    
                    # Blend with original
                    blend_mask = np.zeros_like(overlay)
                    cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0, blend_mask)
                    overlay[mask] = blend_mask[mask]
                    
                    # Add border for overlapping objects
                    if border_color:
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, border_color, 4)  # Thicker border
                    
                    # Add object label with overlap indicator
                    if centroids.get(obj_id):
                        cx, cy = centroids[obj_id]
                        obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                        
                        if is_overlapping:
                            if obj_id in target_objects:
                                # This is a target that's overlapping
                                overlapping_names = []
                                if frame_idx in frame_overlaps and obj_id in frame_overlaps[frame_idx]:
                                    for ov_id in frame_overlaps[frame_idx][obj_id]:
                                        ov_name = object_names.get(ov_id, f"Object_{ov_id}")
                                        overlapping_names.append(ov_name)
                                
                                if overlapping_names:
                                    label = f"🎯{obj_name} ↔ {', '.join(overlapping_names)}"
                                else:
                                    label = f"🎯{obj_name} (overlapping)"
                            else:
                                # This is a regular object overlapping with targets
                                target_names = []
                                for target_id in overlap_targets:
                                    if target_id in target_objects:
                                        target_names.append(target_objects[target_id])
                                
                                if target_names:
                                    label = f"{obj_name} ↔ 🎯{', '.join(target_names)}"
                                else:
                                    label = f"{obj_name} (overlapping)"
                        else:
                            # Add target indicator for non-overlapping targets
                            if obj_id in target_objects:
                                label = f"🎯{obj_name}"
                            else:
                                label = obj_name
                        
                        # Add background for better text visibility
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        label_x = cx - video_x - text_size[0]//2
                        label_y = cy - video_y
                        
                        # Different background color for overlapping objects
                        bg_color = (0, 0, 100) if is_overlapping else (0, 0, 0)
                        cv2.rectangle(overlay, (label_x - 5, label_y - 20), 
                                    (label_x + text_size[0] + 5, label_y + 5), 
                                    bg_color, -1)
                        
                        text_color = (0, 255, 255) if is_overlapping else (255, 255, 255)
                        cv2.putText(overlay, label, (label_x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Place video in center
            output_frame[video_y:video_y+height, video_x:video_x+width] = overlay
            
            # Create and place plots for each object with CONNECTION LINES
            plot_idx = 0
            for obj_id, data in time_series_data.items():
                if not data['frames']:  # Skip empty data
                    continue
                
                plot_color = data['plot_color']
                color_rgb = tuple(int(c * 255) for c in plot_color)
                
                # Get object name and status
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                is_target = data['is_target']
                
                y_offset = plot_idx * side_plot_height
                
                # LEFT PLOT: Movement with CONNECTION LINE
                try:
                    fig_left = Figure(figsize=(side_plot_width/100, side_plot_height/100), dpi=100)
                    ax_left = fig_left.add_subplot(111)
                    
                    if len(data['movement']) > 0 and len(data['frames']) > 0:
                        # Different style for targets
                        line_style = '-' if not is_target else '--'
                        line_width = 2 if not is_target else 3
                        
                        ax_left.plot(data['frames'], data['movement'], color=plot_color, alpha=0.5, linestyle=line_style)
                        ax_left.plot(data['frames'], data['movement_ma'], color=plot_color, linewidth=line_width, linestyle=line_style)
                        ax_left.set_xlim(0, max_frame_idx)
                        ax_left.axvline(frame_idx, color='k', linestyle='--', alpha=0.5)
                        
                        title = f'Movement ({obj_name})'
                        if is_target:
                            title = f'🎯 {title}'
                        ax_left.set_title(title, fontsize=8)
                        ax_left.tick_params(labelsize=6)
                    
                    fig_left.tight_layout()
                    
                    canvas = FigureCanvasAgg(fig_left)
                    canvas.draw()
                    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
                    
                    if y_offset + side_plot_height <= out_height:
                        plot_height = min(side_plot_height, plot_img.shape[0])
                        plot_width = min(side_plot_width, plot_img.shape[1])
                        output_frame[y_offset:y_offset+plot_height, :plot_width] = plot_img[:plot_height, :plot_width]
                    
                    # Draw connection line from left plot to mask centroid
                    if obj_id in centroids and frame_idx < len(data['frames']) and data['frames']:
                        current_data_idx = None
                        for i, f in enumerate(data['frames']):
                            if f <= frame_idx:
                                current_data_idx = i
                        
                        if current_data_idx is not None:
                            value = data['movement_ma'][current_data_idx]
                            
                            y_range = ax_left.get_ylim()
                            if y_range[1] > y_range[0]:
                                plot_height_actual = side_plot_height
                                y_plot = plot_height_actual - ((value - y_range[0]) / (y_range[1] - y_range[0]) * plot_height_actual)
                                y_plot = max(0, min(plot_height_actual, int(y_offset + y_plot)))
                                
                                # Enhanced connection line for targets
                                line_thickness = 3 if is_target else 2
                                start_point = (side_plot_width-2, y_plot)
                                end_point = centroids[obj_id]
                                cv2.line(output_frame, start_point, end_point, color_rgb, line_thickness, cv2.LINE_AA)
                                
                                # Enhanced circle for targets
                                circle_radius = 4 if is_target else 3
                                cv2.circle(output_frame, start_point, circle_radius, color_rgb, -1)
                    
                    plt.close(fig_left)
                    
                except Exception as e:
                    print(f"Error creating movement plot for object {obj_id}: {e}")
                
                # RIGHT PLOT: Area with CONNECTION LINE (similar enhancements)
                try:
                    fig_right = Figure(figsize=(side_plot_width/100, side_plot_height/100), dpi=100)
                    ax_right = fig_right.add_subplot(111)
                    
                    if len(data['areas']) > 0 and len(data['frames']) > 0:
                        line_style = '-' if not is_target else '--'
                        line_width = 2 if not is_target else 3
                        
                        ax_right.plot(data['frames'], data['areas'], color=plot_color, alpha=0.5, linestyle=line_style)
                        ax_right.plot(data['frames'], data['area_ma'], color=plot_color, linewidth=line_width, linestyle=line_style)
                        ax_right.set_xlim(0, max_frame_idx)
                        ax_right.axvline(frame_idx, color='k', linestyle='--', alpha=0.5)
                        
                        title = f'Area ({obj_name})'
                        if is_target:
                            title = f'🎯 {title}'
                        ax_right.set_title(title, fontsize=8)
                        ax_right.tick_params(labelsize=6)
                    
                    fig_right.tight_layout()
                    
                    canvas = FigureCanvasAgg(fig_right)
                    canvas.draw()
                    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
                    
                    x_offset = video_x + width
                    if y_offset + side_plot_height <= out_height and x_offset + side_plot_width <= out_width:
                        plot_height = min(side_plot_height, plot_img.shape[0])
                        plot_width = min(side_plot_width, plot_img.shape[1])
                        output_frame[y_offset:y_offset+plot_height, 
                                    x_offset:x_offset+plot_width] = plot_img[:plot_height, :plot_width]
                    
                    # Draw connection line from right plot to mask centroid
                    if obj_id in centroids and frame_idx < len(data['frames']) and data['frames']:
                        current_data_idx = None
                        for i, f in enumerate(data['frames']):
                            if f <= frame_idx:
                                current_data_idx = i
                        
                        if current_data_idx is not None:
                            value = data['area_ma'][current_data_idx]
                            
                            y_range = ax_right.get_ylim()
                            if y_range[1] > y_range[0]:
                                plot_height_actual = side_plot_height
                                y_plot = plot_height_actual - ((value - y_range[0]) / (y_range[1] - y_range[0]) * plot_height_actual)
                                y_plot = max(0, min(plot_height_actual, int(y_offset + y_plot)))
                                
                                line_thickness = 3 if is_target else 2
                                start_point = (x_offset + 2, y_plot)
                                end_point = centroids[obj_id]
                                cv2.line(output_frame, start_point, end_point, color_rgb, line_thickness, cv2.LINE_AA)
                                
                                circle_radius = 4 if is_target else 3
                                cv2.circle(output_frame, start_point, circle_radius, color_rgb, -1)
                    
                    plt.close(fig_right)
                    
                except Exception as e:
                    print(f"Error creating area plot for object {obj_id}: {e}")
                
                plot_idx += 1
            
            # Add frame info and overlap status - ENHANCED
            info_text = f"Frame {frame_idx}/{len(self.frame_names)-1}"
            if has_overlaps:
                info_text += " - 🎯 OVERLAP DETECTED!"
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Add blinking effect by adding a background
                cv2.rectangle(output_frame, (5, 5), (len(info_text) * 20, 40), (0, 0, 100), -1)
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(output_frame)
        
        out.release()
        print(f"✅ Enhanced analysis video with target overlap detection saved to: {output_path}")
        print(f"📊 Found overlaps in {overlap_frame_count} frames out of {len(self.frame_names)} total frames")
        
        # Print final summary
        if overlap_tracker and overlap_tracker.has_targets():
            summary = overlap_tracker.get_overlap_summary()
            print("\n🎯 Target Overlap Summary:")
            for target_name, data in summary.items():
                print(f"  • {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames")
            print(f"📄 ELAN file will be generated for behavioral analysis")

    def create_simple_analysis_video(self, results, output_path, fps=30, alpha=0.5):
        """Create a simple analysis video without complex plots as fallback"""
        if not results:
            print("No results to analyze!")
            return
            
        print("Creating simple analysis video...")
        
        # Just create a basic video with overlays and object names
        self.save_results_video(results, output_path, fps, show_original=True, alpha=alpha)
        print(f"Simple analysis video saved to: {output_path}")

    def save_results_with_elan(self, results, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results including ELAN file if targets are present"""
        # Save regular results
        self.save_results_video(results, output_path, fps, show_original, alpha)
        self._save_coco_annotations(os.path.join(os.path.dirname(output_path), "segmentation_coco.json"))
        self._save_time_series(os.path.join(os.path.dirname(output_path), "time_series_metrics.csv"))
        
        # Save ELAN file if targets are present
        if hasattr(self, 'overlap_tracker') and self.overlap_tracker.has_targets():
            elan_path = os.path.join(os.path.dirname(output_path), "target_overlaps.eaf")
            object_names = getattr(self, 'object_names', {})
            
            create_elan_file_with_targets(
                video_path=output_path,  # Reference the masked video
                overlap_tracker=self.overlap_tracker,
                output_path=elan_path,
                fps=fps,
                object_names=object_names
            )
            
            print(f"📄 ELAN file saved: {os.path.basename(elan_path)}")

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
        instructions_height = 220  # Increased height for target instructions
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
            "Enter: Finish  Q: Quit",
            "",
            "TARGET TIP: Name objects 'target_1', 'target_2'"
        ]
        
        y_start = height - instructions_height
        for i, instruction in enumerate(instructions):
            if instruction == "":
                continue
            color = (0, 255, 255) if i == 0 else (0, 255, 0) if "TARGET" in instruction else (255, 255, 255)
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
                                     f"Enter name for object {current_obj_id}:\n\n"
                                     f"💡 TIP: Use 'target_1', 'target_2', etc. for crosshairs/targets\n"
                                     f"This will enable automatic overlap detection!",
                                     initialvalue=current_name)
        root.destroy()
        
        if name and name.strip():
            object_names[current_obj_id] = name.strip()
            print(f"Object {current_obj_id} named: {object_names[current_obj_id]}")
            
            # Check if it's a target
            if 'target' in name.lower():
                print(f"🎯 Detected target object: {name}")
            
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
    print("\n🎯 TIP: Name crosshairs/targets as 'target_1', 'target_2', etc.")
    print("   This will automatically track overlaps with other objects!")
    
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

# [Rest of the code continues with the same training and interface classes as in your original script...]
# ... (include all the training classes, fine-tuned inference processor, GUI classes, etc. from your original script)

def run_sam2_training(config_path, video_folder, status_callback=None):
    """Real lightweight training using DeepLabV3+ - ACTUAL TRAINING"""
    try:
        if status_callback:
            status_callback("Loading training configuration...")
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("DeepLabV3+ Fine-tuning Training Started")
        print("=" * 40)
        print(f"Config: {config_path}")
        print(f"Objects: {config['dataset']['object_names']}")
        print(f"Datasets: {len(config['dataset']['annotation_dirs'])}")
        
        if status_callback:
            status_callback("Checking PyTorch availability...")
        
        # Check if training components are available
        if not PYTORCH_TRAINING_AVAILABLE:
            error_msg = "❌ PyTorch training components not available. Please install: pip install torch torchvision"
            print(error_msg)
            if status_callback:
                status_callback("Error: PyTorch training not available")
            return False, error_msg
        
        print("✅ PyTorch training components available")
        
        # Check device
        device = setup_device()
        print(f"Using device: {device}")
        
        if status_callback:
            status_callback("Creating dataset...")
        
        # Create dataset from COCO annotations
        dataset = LightweightSegmentationDataset(
            video_folder=video_folder,
            annotation_dirs=config['dataset']['annotation_dirs'],
            object_names=config['dataset']['object_names'],
            image_size=tuple(config['dataset']['image_size'])
        )
        
        if len(dataset) == 0:
            raise ValueError("No training data found!")
        
        print(f"Created dataset with {len(dataset)} samples")
        
        # Split dataset
        train_size = int(config['dataset']['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        if SKLEARN_AVAILABLE:
            # Use sklearn for better splitting
            indices = list(range(len(dataset)))
            train_indices, val_indices = train_test_split(
                indices, train_size=train_size, random_state=42
            )
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
        else:
            # Use simple random split
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        
        # Create data loaders with proper type conversion and system compatibility
        batch_size = int(config['training']['batch_size'])
        
        # Use 0 workers on Windows to avoid multiprocessing issues
        num_workers = 0 if sys.platform.startswith('win') else 2
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        if status_callback:
            status_callback("Initializing model...")
        
        # Initialize DeepLabV3+ model with proper weights enum
        num_classes = len(config['dataset']['object_names']) + 1  # +1 for background
        
        try:
            # Try to use the modern weights enum
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=21)
        except ImportError:
            # Fallback for older torchvision versions
            model = deeplabv3_resnet50(pretrained=True, num_classes=21)
        
        # Modify classifier for our classes
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model = model.to(device)
        
        print(f"Model initialized for {num_classes} classes (including background)")
        
        # Setup training components with proper type conversion
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
        batch_size = int(config['training']['batch_size'])
        num_epochs = int(config['training']['num_epochs'])
        
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs
        )
        
        # Create output directory
        output_dir = os.path.join(video_folder, config['output']['save_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training settings:")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Weight decay: {weight_decay}")
        print(f"  • Device: {device}")
        
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        
        # Convert evaluation settings
        save_every = int(config['output']['save_every'])
        eval_every = int(config['output']['eval_every'])
        
        for epoch in range(num_epochs):
            if status_callback:
                status_callback(f"Training epoch {epoch+1}/{num_epochs}...")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_samples += images.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            if (epoch + 1) % eval_every == 0:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        outputs = model(images)['out']
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, "deeplabv3_finetuned_best.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'config': config,
                        'object_names': config['dataset']['object_names'],
                        'num_classes': num_classes,
                        'model_type': 'deeplabv3_resnet50'
                    }, best_model_path)
                    print(f"  💾 Saved best model: {best_model_path}")
                
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)
            else:
                training_history['train_loss'].append(avg_train_loss)
            
            scheduler.step()
            
            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"deeplabv3_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'object_names': config['dataset']['object_names'],
                    'num_classes': num_classes,
                    'model_type': 'deeplabv3_resnet50'
                }, checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(output_dir, "deeplabv3_finetuned_final.pth")
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'object_names': config['dataset']['object_names'],
            'num_classes': num_classes,
            'model_type': 'deeplabv3_resnet50',
            'training_history': training_history
        }, final_model_path)
        
        # Save training history plot
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['train_loss'], label='Training Loss')
            plt.plot(training_history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, "training_progress.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"📊 Training plot saved: {plot_path}")
        except Exception as e:
            print(f"Could not save training plot: {e}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Best model: {best_model_path}")
        print(f"📊 Final training loss: {avg_train_loss:.4f}")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
        print(f"🎯 Trained on {len(config['dataset']['object_names'])} object types")
        
        if status_callback:
            status_callback("Training completed successfully!")
        
        return True, best_model_path
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        if status_callback:
            status_callback(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, error_msg

class LightweightSegmentationDataset(Dataset):
    """Dataset for training lightweight segmentation model on SAM2 annotations"""
    
    def __init__(self, video_folder, annotation_dirs, object_names, image_size=(512, 512)):
        self.video_folder = video_folder
        self.object_names = object_names
        self.image_size = image_size
        self.samples = []
        
        # Data augmentation transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        print("Loading training data from COCO annotations...")
        self._load_samples(annotation_dirs)
        print(f"Loaded {len(self.samples)} training samples")
    
    def _load_samples(self, annotation_dirs):
        """Load samples from COCO annotation files"""
        for ann_dir_rel in annotation_dirs:
            ann_dir = os.path.join(self.video_folder, ann_dir_rel)
            coco_file = os.path.join(ann_dir, "segmentation_coco.json")
            
            if not os.path.exists(coco_file):
                print(f"Warning: {coco_file} not found")
                continue
            
            print(f"Loading from: {ann_dir}")
            
            # Load COCO data
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Create mappings
            images_dict = {img['id']: img for img in coco_data['images']}
            
            # Group annotations by image
            image_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
            
            # Process each image with annotations
            for img_id, annotations in image_annotations.items():
                if img_id not in images_dict:
                    continue
                
                image_info = images_dict[img_id]
                image_path = os.path.join(ann_dir, image_info['file_name'])
                
                if not os.path.exists(image_path):
                    continue
                
                self.samples.append({
                    'image_path': image_path,
                    'annotations': annotations,
                    'image_info': image_info
                })
    
    def _create_mask_from_annotations(self, annotations, height, width):
        """Create segmentation mask from COCO annotations"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            # Get object name and map to class ID
            category_name = ann.get('object_name', 'Unknown')
            if category_name in self.object_names:
                class_id = self.object_names.index(category_name) + 1  # +1 for background
            else:
                class_id = 1  # Default to first object class
            
            # Convert segmentation to mask
            try:
                segmentation = ann['segmentation'][0]
                if len(segmentation) >= 6:  # At least 3 points
                    poly = np.array(segmentation).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], class_id)
            except Exception as e:
                print(f"Warning: Could not process annotation: {e}")
                continue
        
        return mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask
        height, width = image.shape[:2]
        mask = self._create_mask_from_annotations(
            sample['annotations'], height, width
        )
        
        # Apply transforms
        image = self.transform(image)
        mask = torch.from_numpy(mask).long()
        
        # Resize mask to match image
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), 
            size=self.image_size, 
            mode='nearest'
        ).squeeze().long()
        
        return image, mask

def create_finetuning_setup(video_folder):
    """Create fine-tuning setup files in the video folder with auto-detection - ENHANCED"""
    try:
        # Scan for existing annotation files
        annotation_dirs = []
        object_names_set = set()
        
        print(f"Scanning {video_folder} for annotation files...")
        
        for item in os.listdir(video_folder):
            item_path = os.path.join(video_folder, item)
            if os.path.isdir(item_path) and item.endswith('_frames'):
                # Check if this directory has segmentation files
                coco_file = os.path.join(item_path, "segmentation_coco.json")
                csv_file = os.path.join(item_path, "time_series_metrics.csv")
                
                if os.path.exists(coco_file) and os.path.exists(csv_file):
                    annotation_dirs.append(item_path)
                    print(f"  Found annotations: {item}")
                    
                    # Extract object names from COCO file
                    try:
                        with open(coco_file, 'r', encoding='utf-8') as f:
                            coco_data = json.load(f)
                        
                        for category in coco_data.get('categories', []):
                            obj_name = category.get('name', 'Unknown')
                            if obj_name and not obj_name.startswith('Object_'):
                                object_names_set.add(obj_name)
                    except Exception as e:
                        print(f"    Warning: Could not read object names from {coco_file}: {e}")
        
        if not annotation_dirs:
            raise ValueError("No annotation files found! Please process some videos first.")
        
        # Convert to relative paths for portability
        annotation_dirs_relative = [os.path.relpath(d, video_folder) for d in annotation_dirs]
        object_names_list = list(object_names_set) if object_names_set else ["Object_1", "Object_2"]
        
        print(f"Found {len(annotation_dirs)} annotated video folders")
        print(f"Detected object names: {object_names_list}")
        
        # Create training config
        config_content = f"""# DeepLabV3+ Fine-tuning Configuration - Auto-generated
model:
  type: "deeplabv3_resnet50"  # Lightweight segmentation model
  pretrained: true  # Use ImageNet pretrained weights
  
dataset:
  name: "custom_objects"
  annotation_dirs: {annotation_dirs_relative}  # Auto-detected from folder
  object_names: {object_names_list}  # Auto-detected from annotations
  train_split: 0.8
  val_split: 0.2
  image_size: [512, 512]  # Reduced for faster training
  
training:
  batch_size: 4  # Reasonable for most GPUs
  learning_rate: 1e-4  # Good for fine-tuning
  num_epochs: 15  # Quick training for testing
  weight_decay: 1e-4
  
output:
  save_dir: "sam2_finetuned"  # Will be created in this folder
  save_every: 5
  eval_every: 2

# Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Base folder: {video_folder}
# Found {len(annotation_dirs)} training datasets
# Model: DeepLabV3+ ResNet50 (lightweight segmentation)
# Training approach: Fine-tune on your SAM2 annotations
"""

        # Write config to video folder
        config_path = os.path.join(video_folder, "training_config.yaml")
        with open(config_path, "w", encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ Created: {config_path}")
        
        return True, len(annotation_dirs), object_names_list, config_path
        
    except Exception as e:
        print(f"❌ Error creating fine-tuning setup: {e}")
        return False, 0, [], None

class VideoAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAM2 Video Analysis - Enhanced with Target Overlap Detection")
        self.root.geometry("700x850")  # Increased height for new options
        self.root.minsize(700, 850)
        
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
        """Setup the GUI with target overlap detection options"""
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="SAM2 Video Analysis - Enhanced with Target Overlap Detection", 
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
        
        # Chunk size setting
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
        
        # Target overlap detection section - NEW
        overlap_frame = tk.LabelFrame(main_frame, text="🎯 Target Overlap Detection", font=("Arial", 9, "bold"))
        overlap_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Info label
        info_label = tk.Label(overlap_frame, 
                             text='💡 Objects named "target_1", "target_2", etc. will be tracked for overlaps with other objects',
                             font=("Arial", 8), fg="darkgreen", wraplength=650)
        info_label.pack(anchor=tk.W, padx=5, pady=(5, 2))
        
        # Overlap threshold setting
        threshold_frame = tk.Frame(overlap_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threshold_frame, text="Overlap Threshold (%):").pack(side=tk.LEFT)
        self.overlap_threshold_var = tk.StringVar(value="10")
        threshold_spin = tk.Spinbox(threshold_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.overlap_threshold_var, width=10)
        threshold_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(threshold_frame, text="(minimum overlap to register event)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Enable/disable overlap tracking
        self.enable_overlap_tracking = tk.BooleanVar(value=True)
        overlap_cb = tk.Checkbutton(overlap_frame, 
                                   text="📊 Export ELAN file for behavioral analysis",
                                   variable=self.enable_overlap_tracking)
        overlap_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Example text
        example_label = tk.Label(overlap_frame, 
                                text='Example: Name objects "target_1", "apple", "hand" → ELAN will show when target_1 overlaps with apple or hand',
                                font=("Arial", 8), fg="blue", wraplength=650)
        example_label.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        # Fine-tuning section
        finetuning_frame = tk.LabelFrame(main_frame, text="🧠 Fine-tuning Workflow", font=("Arial", 9, "bold"))
        finetuning_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Setup button
        setup_button = tk.Button(finetuning_frame, text="🔧 Setup Fine-tuning Environment", 
                 command=self.setup_finetuning, bg="#FF5722", fg="white",
                 font=("Arial", 9))
        setup_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Train button (initially disabled)
        self.train_button = tk.Button(finetuning_frame, text="🚀 Start Training Model", 
                 command=self.start_training, bg="#3F51B5", fg="white",
                 font=("Arial", 9), state="disabled")
        self.train_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Training status
        self.training_status_var = tk.StringVar(value="")
        self.training_status_label = tk.Label(finetuning_frame, textvariable=self.training_status_var, 
                                             font=("Arial", 8), fg="blue")
        self.training_status_label.pack(pady=2)
        
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
        """Setup fine-tuning environment"""
        try:
            folder = self.folder_var.get()
            if not folder:
                messagebox.showwarning("Warning", "Please select a video folder first")
                return
            
            self.status_var.set("Setting up fine-tuning environment...")
            self.root.update()
            
            success, num_datasets, object_names, config_path = create_finetuning_setup(folder)
            
            if success:
                self.training_status_var.set(f"Setup complete: {num_datasets} datasets, {len(object_names)} object types")
                
                # Enable train button
                self.train_button.config(state="normal")
                
                messagebox.showinfo("Setup Complete", 
                    f"Fine-tuning environment setup complete!\n\n"
                    f"📊 Found {num_datasets} annotated video folders\n"
                    f"🎯 Detected objects: {', '.join(object_names)}\n"
                    f"📝 Config saved: {os.path.basename(config_path)}\n\n"
                    f"✅ Ready to start training!")
                
                self.status_var.set("Fine-tuning setup complete - ready to train!")
            else:
                messagebox.showerror("Setup Failed", 
                    "Fine-tuning setup failed. Make sure you have processed some videos first.")
                self.status_var.set("Fine-tuning setup failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup fine-tuning: {str(e)}")
            self.status_var.set("Setup failed")
    
    def start_training(self):
        """Start training fine-tuned model"""
        try:
            folder = self.folder_var.get()
            if not folder:
                messagebox.showwarning("Warning", "Please select a video folder first")
                return
            
            config_path = os.path.join(folder, "training_config.yaml")
            if not os.path.exists(config_path):
                messagebox.showerror("Error", 
                    "Training configuration not found. Please run setup first.")
                return
            
            # Confirm training start
            result = messagebox.askyesno("Start Training", 
                "Start SAM2 fine-tuning training?\n\n"
                "This will:\n"
                "• Use all annotated videos in this folder\n"
                "• Create a specialized model for your objects\n"
                "• Take some time to complete\n\n"
                "Continue?")
            
            if not result:
                return
            
            self.status_var.set("Training in progress...")
            self.train_button.config(state="disabled")
            self.root.update()
            
            def status_update(message):
                self.training_status_var.set(message)
                self.root.update()
            
            # Run training
            success, model_path = run_sam2_training(config_path, folder, status_update)
            
            if success:
                self.training_status_var.set("Training completed successfully!")
                
                messagebox.showinfo("Training Complete", 
                    f"🎉 Fine-tuning completed successfully!\n\n"
                    f"📁 Model saved: {os.path.basename(model_path)}\n"
                    f"🎯 Ready for inference on new videos!\n\n"
                    f"✅ You can now use the trained model for automatic detection")
                
                self.status_var.set("Training completed - model ready for inference!")
            else:
                self.training_status_var.set("Training failed")
                messagebox.showerror("Training Failed", f"Training failed: {model_path}")
                self.status_var.set("Training failed")
            
            self.train_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.training_status_var.set("Training failed")
            self.train_button.config(state="normal")
            self.status_var.set("Training failed")
    
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
        """Get frame number with preview functionality"""
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
        """Process the selected video with enhanced target overlap detection"""
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
            
            # Get frame number with preview
            frame_num = self.get_frame_number_with_preview(frames_dir, num_frames)
            if frame_num is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Get settings from UI
            try:
                chunk_size = int(self.chunk_size_var.get())
                overlap_threshold = float(self.overlap_threshold_var.get()) / 100.0
            except ValueError:
                chunk_size = 500
                overlap_threshold = 0.1
                self.chunk_size_var.set("500")
                self.overlap_threshold_var.set("10")
            
            # Initialize processor with overlap tracking
            self.status_var.set("Initializing processor...")
            self.root.update()
            
            processor = VideoChunkProcessor(
                predictor=self.predictor, 
                video_dir=frames_dir, 
                chunk_size=chunk_size,
                overlap_frames=30,
                interactive_correction=self.interactive_correction.get(),
                seed_frame_idx=frame_num,
                overlap_threshold=overlap_threshold  # Add overlap threshold
            )
            
            # Load the selected frame for mask selection
            frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
            if not os.path.exists(frame_path):
                messagebox.showerror("Error", f"Frame {frame_num} not found")
                return
            
            frame = cv2.imread(frame_path)
            
            # Point selection with target instructions
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
                "🔄 Processing goes both directions from here\n\n"
                "🎯 TARGET TIP: Name crosshairs/targets as 'target_1', 'target_2'\n"
                "   This will automatically track overlaps with other objects!")
            
            points_dict, labels_dict, object_names = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Process video with enhanced tracking
            self.status_var.set(f"🔄 Processing video with target overlap detection...")
            self.root.update()
            
            # IMPORTANT: Set object names on processor BEFORE processing
            processor.object_names = object_names
            
            results = processor.process_video(points_dict, labels_dict)
            
            if results:
                # Ensure object names are set for ELAN export
                processor.results = results
                processor.object_names = object_names
                
                # Enhanced saving with ELAN export
                self.status_var.set("Saving results...")
                self.root.update()
                
                output_path = os.path.join(frames_dir, "output_masked.mp4")
                
                if self.enable_overlap_tracking.get() and hasattr(processor, 'overlap_tracker'):
                    print("🎯 Saving results with ELAN export...")
                    processor.save_results_with_elan(
                        results=results,
                        output_path=output_path,
                        fps=fps,
                        show_original=True,
                        alpha=0.5
                    )
                else:
                    print("📄 Saving standard results...")
                    processor.save_results(
                        output_path=output_path,
                        fps=fps,
                        show_original=True,
                        alpha=0.5
                    )
                
                # Create analysis video if requested
                if self.create_analysis_video.get():
                    self.status_var.set("Creating analysis video...")
                    self.root.update()
                    
                    analysis_output = os.path.join(frames_dir, "analysis_video.mp4")
                    try:
                        processor.create_analysis_video(
                            results=results,
                            output_path=analysis_output,
                            fps=fps,
                            alpha=0.5
                        )
                        analysis_created = True
                        print("✅ Enhanced analysis video created successfully")
                        
                    except Exception as e:
                        print(f"⚠️ Enhanced analysis video failed: {e}")
                        try:
                            simple_output = os.path.join(frames_dir, "simple_analysis_video.mp4")
                            processor.create_simple_analysis_video(
                                results=results,
                                output_path=simple_output,
                                fps=fps,
                                alpha=0.5
                            )
                            analysis_created = True
                            print("✅ Simple analysis video created successfully")
                        except Exception as e2:
                            print(f"❌ Simple analysis video also failed: {e2}")
                            analysis_created = False
                else:
                    analysis_created = False
                
                self.status_var.set("Processing completed!")
                
                # Enhanced success message
                target_info = ""
                if hasattr(processor, 'overlap_tracker') and processor.overlap_tracker.has_targets():
                    summary = processor.overlap_tracker.get_overlap_summary()
                    target_info = f"\n\n🎯 Target Overlap Events:\n"
                    for target_name, data in summary.items():
                        target_info += f"  • {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames\n"
                    target_info += "\n📄 ELAN file: target_overlaps.eaf"
                
                named_objects = [name for name in object_names.values()]
                objects_summary = "\n".join([f"  • {name}" for name in named_objects])
                
                success_msg = f"""🎉 Processing Complete with Target Overlap Detection!

Reference Frame: {frame_num}
Chunk Size: {chunk_size} frames
Overlap Threshold: {overlap_threshold*100:.1f}%
Results saved in: {frames_dir}

📁 Generated Files:
• output_masked.mp4 - Video with overlays
• segmentation_coco.json - Annotations
• time_series_metrics.csv - Movement data"""

                if analysis_created:
                    if os.path.exists(os.path.join(frames_dir, "analysis_video.mp4")):
                        success_msg += "\n• analysis_video.mp4 - Enhanced analysis with connection lines"
                    elif os.path.exists(os.path.join(frames_dir, "simple_analysis_video.mp4")):
                        success_msg += "\n• simple_analysis_video.mp4 - Simple analysis video"

                success_msg += f"""{target_info}

📊 Processed Objects ({len(object_names)}):
{objects_summary}

✅ Enhanced processing with target overlap detection completed!"""
                
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
    print("Starting SAM2 Video Analysis - Enhanced with Target Overlap Detection!")
    print("=" * 70)
    
    print("\n✨ NEW FEATURES:")
    print("  🎯 Target Overlap Detection - Track when targets overlap with objects")
    print("  📄 ELAN Export - Behavioral analysis file generation")
    print("  📊 Enhanced Analysis Video - Connection lines + overlap highlighting")
    print("  ⚙️ Configurable Overlap Threshold - Adjust sensitivity")
    print()
    print("🎯 TARGET WORKFLOW:")
    print("  1. 📹 Process video → Name crosshairs/targets as 'target_1', 'target_2'")
    print("  2. 🔄 Automatic overlap tracking → Detects when targets overlap with objects")
    print("  3. 📄 ELAN export → target_overlaps.eaf for behavioral analysis")
    print("  4. 📊 Enhanced visualization → Connection lines + overlap highlighting")
    print()
    print("📋 PERFECT FOR:")
    print("  • Behavioral Analysis - Human-computer interaction studies")
    print("  • Gaze Tracking - Eye-tracking research with fixation points")
    print("  • Tool Interaction - Hand-tool coordination analysis")
    print("  • Spatial Cognition - Object-target relationship studies")
    print()
    print("📦 Requirements:")
    print("  • SAM2 installed and checkpoints downloaded")
    print("  • FFmpeg for video processing")
    print("  • Python packages: opencv-python, torch, matplotlib, pandas, tqdm")
    print()
    print("🎬 Starting enhanced application...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()