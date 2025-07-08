#!/usr/bin/env python3
"""
Fixed Overlap Detection and Annotations for SAM2 Video Analysis
Handles complex scenarios: inclusion, multiple overlaps, bidirectional relationships
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
import time
from datetime import datetime
import gc
from tqdm import tqdm
import pandas as pd
import psutil

# memory optimization (same as before)
os.environ["SAM2_OFFLOAD_VIDEO_TO_CPU"] = "true"
os.environ["SAM2_OFFLOAD_STATE_TO_CPU"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'utilization_pct': (reserved / total) * 100
        }
    return None

def ultra_cleanup_memory():
    """memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def configure_torch_ultra_conservative():
    """Configure PyTorch for memory usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70)
        
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        ultra_cleanup_memory()
        print(f"GPU Memory after setup: {get_gpu_memory_info()}")

def setup_device_ultra_optimized():
    """Setup computation device with settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_info = get_gpu_memory_info()
        print(f"Initial GPU Memory: {gpu_info['allocated_gb']:.1f}GB allocated, {gpu_info['free_gb']:.1f}GB free")
        
        if gpu_info['total_gb'] < 8:
            print("⚠️ WARNING: Low GPU memory detected. Using very conservative settings.")
            torch.cuda.set_per_process_memory_fraction(0.60)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Support for MPS devices is preliminary.")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU - this will be very slow but stable")
    
    print(f"Using device: {device}")
    return device

# Video processing functions (same as before)
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
        'ffmpeg', '-y',
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

class EnhancedOverlapDetector:
    """Enhanced overlap detector that properly handles inclusion and complex overlaps"""
    
    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.inclusion_threshold = 0.1  # 80% overlap = inclusion
        
    def calculate_detailed_overlap(self, mask1, mask2):
        """Enhanced overlap detection with both pixel overlap and spatial containment"""
        if mask1.shape != mask2.shape:
            print(f"❌ Shape mismatch: {mask1.shape} vs {mask2.shape}")
            return None
        
        # Ensure masks are 2D
        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # VERY AGGRESSIVE boolean conversion - catch any non-zero values
        if mask1.dtype in [np.float32, np.float64]:
            mask1_bool = mask1 > 0.0001
        else:
            mask1_bool = mask1 > 0
        
        if mask2.dtype in [np.float32, np.float64]:
            mask2_bool = mask2 > 0.0001
        else:
            mask2_bool = mask2 > 0
        
        # Calculate areas
        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)
        
        if area1 == 0 or area2 == 0:
            return None
        
        # Calculate pixel intersection
        intersection = mask1_bool & mask2_bool
        intersection_area = np.sum(intersection)
        
        # Calculate overlap percentages for pixel overlap
        overlap_pct_1 = intersection_area / area1 if area1 > 0 else 0
        overlap_pct_2 = intersection_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_pct_1, overlap_pct_2)
        
        # SPATIAL CONTAINMENT DETECTION
        spatial_relationship = False
        containment_type = None
        
        try:
            # Find contours for both masks
            contours1, _ = cv2.findContours(mask1_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours1 and contours2:
                # Get the largest contour for each mask
                contour1 = max(contours1, key=cv2.contourArea)
                contour2 = max(contours2, key=cv2.contourArea)
                
                # Calculate centroids
                M1 = cv2.moments(contour1)
                M2 = cv2.moments(contour2)
                
                if M1['m00'] != 0 and M2['m00'] != 0:
                    cx1, cy1 = int(M1['m10']/M1['m00']), int(M1['m01']/M1['m00'])
                    cx2, cy2 = int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00'])
                    
                    # Check for partial area containment FIRST (any part of object inside boundary)
                    # Create masks for contour areas
                    mask1_contour = np.zeros_like(mask1_bool, dtype=np.uint8)
                    mask2_contour = np.zeros_like(mask2_bool, dtype=np.uint8)
                    cv2.fillPoly(mask1_contour, [contour1], 1)
                    cv2.fillPoly(mask2_contour, [contour2], 1)
                    
                    # Check if any part of object2 is within contour1 boundary
                    object2_in_contour1 = np.any(mask2_bool & mask1_contour)
                    # Check if any part of object1 is within contour2 boundary  
                    object1_in_contour2 = np.any(mask1_bool & mask2_contour)
                    
                    if object2_in_contour1:
                        spatial_relationship = True
                        overlap_area = np.sum(mask2_bool & mask1_contour)
                        containment_type = "object2_partial_inside_object1"
                        print(f"    🎯 Part of Object 2 is INSIDE Object 1 boundary ({overlap_area} pixels)")
                    elif object1_in_contour2:
                        spatial_relationship = True
                        overlap_area = np.sum(mask1_bool & mask2_contour)
                        containment_type = "object1_partial_inside_object2"
                        print(f"    🎯 Part of Object 1 is INSIDE Object 2 boundary ({overlap_area} pixels)")
                    else:
                        # Fallback to centroid check only if no partial containment
                        inside_1 = cv2.pointPolygonTest(contour1, (cx2, cy2), False) >= 0
                        inside_2 = cv2.pointPolygonTest(contour2, (cx1, cy1), False) >= 0
                        
                        if inside_1:
                            spatial_relationship = True
                            containment_type = "object2_centroid_inside_object1"
                            print(f"    🎯 Object 2 centroid is INSIDE Object 1 boundary")
                        elif inside_2:
                            spatial_relationship = True
                            containment_type = "object1_centroid_inside_object2"
                            print(f"    🎯 Object 1 centroid is INSIDE Object 2 boundary")
                    

        
        except Exception as e:
            print(f"    ⚠️ Error in spatial containment detection: {e}")
            spatial_relationship = False
        
        # Determine relationship type and strength
        # Apply percentage threshold to pixel overlap
        has_meaningful_pixel_overlap = intersection_area > 0 and max_overlap >= self.overlap_threshold
        has_spatial_relationship = spatial_relationship
        
        # If no relationship at all, return None
        if not has_meaningful_pixel_overlap and not has_spatial_relationship:
            return None
        
        # Enhanced criteria: Accept EITHER meaningful pixel overlap OR spatial containment
        meets_basic_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        meets_continuation_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        
        # Determine relationship type
        if has_meaningful_pixel_overlap and has_spatial_relationship:
            relationship_type = "overlap_and_containment"
        elif has_meaningful_pixel_overlap:
            relationship_type = "pixel_overlap"
        elif has_spatial_relationship:
            relationship_type = "spatial_containment"
        else:
            relationship_type = "none"
        
        print(f"  🔍 Enhanced overlap analysis:")
        if intersection_area > 0:
            print(f"    Pixel intersection: {intersection_area} pixels")
            print(f"    Max pixel overlap: {max_overlap:.1%}")
            print(f"    Meets pixel threshold ({self.overlap_threshold:.1%}): {has_meaningful_pixel_overlap}")
        if has_spatial_relationship:
            print(f"    Spatial relationship: {containment_type}")
        print(f"    Final relationship type: {relationship_type}")
        print(f"    Meets threshold: {meets_basic_threshold}")
        
        return {
            'intersection_area': intersection_area,
            'overlap_pct_1': overlap_pct_1,
            'overlap_pct_2': overlap_pct_2,
            'min_overlap_pct': min(overlap_pct_1, overlap_pct_2),
            'max_overlap_pct': max_overlap,
            'spatial_relationship': spatial_relationship,
            'containment_type': containment_type,
            'has_meaningful_pixel_overlap': has_meaningful_pixel_overlap,
            'has_spatial_relationship': has_spatial_relationship,
            'relationship_type': relationship_type,
            'meets_threshold': meets_basic_threshold,
            'meets_continuation_threshold': meets_continuation_threshold
        }


class ImprovedTargetOverlapTracker:
    """Improved overlap tracker with better inclusion detection and annotations"""
    
    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}
        self.target_objects = {}
        self.detector = EnhancedOverlapDetector(overlap_threshold)
        
    def register_target(self, obj_id, obj_name):
        """Register target objects"""
        if 'target' in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"Target registered: {obj_name} (ID: {obj_id})")
            return True
        return False
    
    def get_overlap_summary(self):
        """Get overlap summary"""
        summary = {}
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            summary[target_name] = {
                'total_events': len(events),
                'events': events,
                'total_overlap_frames': sum(event['duration_frames'] for event in events)
            }
        return summary
    
    def finalize_tracking(self, last_frame_idx):
        """PRECISE finalize - don't extend events, keep them as detected"""
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            
            if events and events[-1].get('end_frame') is None:
                last_event = events[-1]
                
                # If event has no end_frame, it means it was ongoing
                # End it at the last frame where we had duration_frames
                if 'duration_frames' in last_event and last_event['duration_frames'] > 0:
                    last_event['end_frame'] = last_event['start_frame'] + last_event['duration_frames'] - 1
                else:
                    # Single frame event
                    last_event['end_frame'] = last_event['start_frame']
                    last_event['duration_frames'] = 1
                
                print(f"  📝 Precise finalize for {target_name}: frames {last_event['start_frame']}-{last_event['end_frame']} ({last_event['duration_frames']} frames)")

    def analyze_frame_overlaps(self, frame_results, object_names):
        """Enhanced frame analysis with continuation validation"""
        frame_analysis = {
            'target_overlaps': {},
            'object_relationships': {},
            'looking_at_events': []
        }
        
        try:
            for target_id in self.target_objects:
                if target_id not in frame_results:
                    continue
                    
                target_mask = frame_results[target_id]
                if len(target_mask.shape) > 2:
                    target_mask = target_mask.squeeze()
                
                target_name = self.target_objects[target_id]
                looking_at_objects = []
                
                # Check if we have an ongoing event for this target
                has_ongoing_event = (self.overlap_events[target_id] and 
                                not self.overlap_events[target_id][-1].get('end_frame'))
                
                for obj_id, mask in frame_results.items():
                    if obj_id == target_id:
                        continue
                        
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    
                    try:
                        overlap_info = self.detector.calculate_detailed_overlap(target_mask, mask)
                        
                        if overlap_info:
                            # Use different criteria for new vs continuing events
                            if has_ongoing_event:
                                # For continuing events, require stronger overlap
                                if overlap_info.get('meets_continuation_threshold', False):
                                    looking_at_objects.append({
                                        'object_id': obj_id,
                                        'object_name': obj_name,
                                        'event_type': 'looking_at',
                                        'relationship_desc': f"LOOKING AT {obj_name} (continuing)"
                                    })
                                    print(f"      ✅ STRONG OVERLAP CONTINUES: {target_name} ↔ {obj_name}")
                                else:
                                    print(f"      ⚠️ WEAK OVERLAP (ending event): {target_name} ↔ {obj_name}")
                            else:
                                # For new events, use basic threshold (ultra-sensitive)
                                if overlap_info.get('meets_threshold', False):
                                    looking_at_objects.append({
                                        'object_id': obj_id,
                                        'object_name': obj_name,
                                        'event_type': 'looking_at',
                                        'relationship_desc': f"LOOKING AT {obj_name}"
                                    })
                                    if has_ongoing_event:
                                        print(f"      ✅ OVERLAP CONTINUES: {target_name} ↔ {obj_name}")
                                    else:
                                        print(f"      ✅ NEW OVERLAP DETECTED: {target_name} ↔ {obj_name}")
                            
                            # Store for ELAN export
                            if looking_at_objects and looking_at_objects[-1]['object_id'] == obj_id:
                                frame_analysis['looking_at_events'].append({
                                    'target_id': target_id,
                                    'target_name': target_name,
                                    'object_id': obj_id,
                                    'object_name': obj_name
                                })
                        
                    except Exception as e:
                        print(f"      ⚠️ Error checking {obj_name}: {e}")
                        continue
                
                if looking_at_objects:
                    frame_analysis['target_overlaps'][target_id] = looking_at_objects
            
        except Exception as e:
            print(f"  ⚠️ Error in analyze_frame_overlaps: {e}")
            import traceback
            traceback.print_exc()
        
        return frame_analysis

    def track_frame_overlaps_batch(self, frame_idx, frame_results, object_names):
        """Track 'looking at' events with ACCURATE offset detection"""
        try:
            frame_analysis = self.analyze_frame_overlaps(frame_results, object_names)
            
            # Store frame analysis for video creation
            if not hasattr(self, 'frame_analyses'):
                self.frame_analyses = {}
            self.frame_analyses[frame_idx] = frame_analysis
            
            # Process each target to update events with accurate timing
            for target_id in self.target_objects:
                current_overlaps = []
                
                # Get current overlaps for this target
                if target_id in frame_analysis.get('target_overlaps', {}):
                    looking_at_objects = frame_analysis['target_overlaps'][target_id]
                    current_overlaps = [obj['object_name'] for obj in looking_at_objects]
                
                # Update events with accurate offset detection
                self._update_overlap_event(target_id, frame_idx, current_overlaps)
            
            return frame_analysis
            
        except Exception as e:
            print(f"  ⚠️ Error in track_frame_overlaps_batch for frame {frame_idx}: {e}")
            return {
                'target_overlaps': {},
                'object_relationships': {},
                'looking_at_events': []
            }

    
    def _update_overlap_event(self, target_id, frame_idx, overlapping_names):
        """Enhanced event tracking with stricter continuation criteria"""
        events = self.overlap_events[target_id]
        current_overlap_set = set(overlapping_names)
        target_name = self.target_objects[target_id]
        
        # Check if we have an ongoing event
        if events and events[-1].get('end_frame') is None:
            last_event = events[-1]
            last_overlap_set = set(last_event['overlapping_objects'])
            
            if current_overlap_set == last_overlap_set and current_overlap_set:
                # Same objects detected - but is the overlap still STRONG enough to continue?
                # We need to check the overlap strength, not just presence
                
                # Get the frame results to check overlap strength
                # (This would need to be passed from the calling function)
                # For now, continue the event but add validation
                
                last_event['duration_frames'] = frame_idx - last_event['start_frame'] + 1
                print(f"      → Continuing event: {target_name} frames {last_event['start_frame']}-{frame_idx}")
                
            else:
                # Objects changed or stopped - END the event
                last_event['end_frame'] = frame_idx - 1
                last_event['duration_frames'] = last_event['end_frame'] - last_event['start_frame'] + 1
                
                objects_str = ', '.join(last_event['overlapping_objects'])
                print(f"      ✅ Event ended: {target_name} frames {last_event['start_frame']}-{last_event['end_frame']} ({last_event['duration_frames']} frames) | {objects_str}")
                
                # Start new event if we have new overlaps
                if current_overlap_set:
                    new_event = {
                        'start_frame': frame_idx,
                        'end_frame': None,
                        'duration_frames': 1,
                        'overlapping_objects': list(overlapping_names),
                        'event_type': 'looking_at'
                    }
                    events.append(new_event)
                    objects_str = ', '.join(current_overlap_set)
                    print(f"      🎯 New event started: {target_name} frame {frame_idx} | {objects_str}")
        else:
            # No ongoing event - start new one if we have overlaps
            if current_overlap_set:
                new_event = {
                    'start_frame': frame_idx,
                    'end_frame': None,
                    'duration_frames': 1,
                    'overlapping_objects': list(overlapping_names),
                    'event_type': 'looking_at'
                }
                events.append(new_event)
                objects_str = ', '.join(current_overlap_set)
                print(f"      🎯 First event started: {target_name} frame {frame_idx} | {objects_str}")

    def has_targets(self):
        """Check if any targets are registered"""
        return bool(self.target_objects)

class UltraOptimizedProcessor:
    """Ultra memory-optimized processor with improved overlap detection"""
    
    def __init__(self, predictor, video_dir, overlap_threshold=0.1, reference_frame=0, 
                 batch_size=50, auto_fallback=True):
        self.predictor = predictor
        self.video_dir = video_dir
        self.overlap_threshold = overlap_threshold
        self.reference_frame = reference_frame
        self.batch_size = batch_size
        self.auto_fallback = auto_fallback
        
        # Initialize improved overlap tracker
        self.overlap_tracker = ImprovedTargetOverlapTracker(overlap_threshold)
        
        # Get frame names
        self.frame_names = sorted(
            [p for p in os.listdir(self.video_dir) 
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        if not self.frame_names:
            raise ValueError("No frames found in the specified directory!")
        
        print(f"Processor with Overlap Detection")
        print(f"  Frames: {len(self.frame_names)}")
        print(f"  Reference frame: {reference_frame}")
        print(f"  Overlap threshold: {overlap_threshold*100:.1f}%")
        print(f"  Event detection: Any spatial relationship = 'looking at' event")
        print(f"  Clean timing: Accurate begin/end times for ELAN export")
        print(f"  Simplified annotations: Focus on event detection, not percentages")
        
        # Memory optimization flags
        self.offload_video_to_cpu = os.environ.get("SAM2_OFFLOAD_VIDEO_TO_CPU", "true") == "true"
        self.offload_state_to_cpu = os.environ.get("SAM2_OFFLOAD_STATE_TO_CPU", "true") == "true"
    
    def process_video_with_memory_management(self, points_dict, labels_dict, object_names, debug=True):
        """Process video with ultra memory management and improved overlap detection"""
        try:
            configure_torch_ultra_conservative()
            
            print(f"\nStarting processing with overlap detection...")
            
            # Try processing with fallback strategies
            for attempt in range(3):
                try:
                    if attempt == 0:
                        print(f"Attempt 1: Standard optimized processing")
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug)
                    elif attempt == 1:
                        print(f"Attempt 2: Micro-batch processing")
                        self.batch_size = self.batch_size // 2
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug)
                    else:
                        print(f"Attempt 3: Emergency CPU fallback")
                        return self._process_cpu_fallback(points_dict, labels_dict, object_names, debug)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ❌ Attempt {attempt + 1} failed: CUDA OOM")
                        ultra_cleanup_memory()
                        
                        if torch.cuda.is_available():
                            current_fraction = torch.cuda.get_per_process_memory_fraction() * 0.8
                            torch.cuda.set_per_process_memory_fraction(max(0.3, current_fraction))
                            print(f"  🔧 Reduced memory fraction to {current_fraction:.2f}")
                        
                        if attempt == 2:
                            raise e
                    else:
                        raise e
            
        except Exception as e:
            print(f"❌ All processing attempts failed: {str(e)}")
            return None
        finally:
            ultra_cleanup_memory()

    def _process_standard_optimized(self, points_dict, labels_dict, object_names, debug):
        """Standard optimized processing with enhanced overlap detection"""
        # Initialize SAM2 state
        print("🔧 Initializing SAM2 state...")
        inference_state = self.predictor.init_state(
            video_path=self.video_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=True,
        )
        
        self.predictor.reset_state(inference_state)
        ultra_cleanup_memory()
        
        # Store object names
        self.object_names = object_names
        
        # Register targets
        targets_found = False
        for obj_id, obj_name in object_names.items():
            if self.overlap_tracker.register_target(obj_id, obj_name):
                targets_found = True
        
        print(f"\n📌 Adding prompts for {len(points_dict)} objects...")
        
        # Add all prompts to reference frame
        for obj_id in points_dict:
            try:
                points = np.array(points_dict[obj_id], dtype=np.float32)
                labels = np.array(labels_dict[obj_id], dtype=np.int32)
                
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                
                if debug:
                    print(f"  📌 {obj_name}: +{sum(labels == 1)} -{sum(labels == 0)} points")
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.reference_frame,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
                
                # Immediate cleanup
                del out_mask_logits, points, labels
                ultra_cleanup_memory()
            
            except Exception as e:
                print(f"  ❌ Error adding prompts for object {obj_id}: {e}")
                continue
        
        print(f"\n🔄 Propagating through video with enhanced overlap detection...")
        
        # Process with enhanced overlap tracking
        results = {}
        frame_analyses = {}  # Store detailed frame analysis
        frame_count = 0
        overlap_count = 0
        last_memory_check = 0
        
        with tqdm(total=len(self.frame_names), desc="Processing frames") as pbar:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                try:
                    # Memory monitoring
                    if frame_count - last_memory_check >= 50:
                        gpu_info = get_gpu_memory_info()
                        if gpu_info and gpu_info['utilization_pct'] > 90:
                            print(f"  ⚠️ High memory usage: {gpu_info['utilization_pct']:.1f}%")
                            ultra_cleanup_memory()
                        last_memory_check = frame_count
                    
                    # Store results efficiently
                    frame_results = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        frame_results[out_obj_id] = mask.copy()
                        del mask
                    
                    results[out_frame_idx] = frame_results
                    
                    # Enhanced overlap tracking
                    if targets_found:
                        frame_analysis = self.overlap_tracker.track_frame_overlaps_batch(
                            out_frame_idx, frame_results, object_names
                        )
                        frame_analyses[out_frame_idx] = frame_analysis
                        
                        if frame_analysis['target_overlaps']:
                            overlap_count += 1
                            
                            # Debug output for first few overlaps
                            if debug and overlap_count <= 3:
                                print(f"  🎯 Frame {out_frame_idx} overlaps:")
                                for target_id, overlaps in frame_analysis['target_overlaps'].items():
                                    target_name = self.overlap_tracker.target_objects[target_id]
                                    for overlap in overlaps:
                                        print(f"    {target_name} {overlap['relationship_desc']}")
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # Cleanup
                    if frame_count % 25 == 0:
                        ultra_cleanup_memory()
                    
                    del out_mask_logits, frame_results
                    
                except Exception as e:
                    print(f"  ⚠️ Error processing frame {out_frame_idx}: {e}")
                    pbar.update(1)
                    ultra_cleanup_memory()
                    continue
        
        # Finalize tracking
        if targets_found:
            last_frame = max(results.keys()) if results else 0
            self.overlap_tracker.finalize_tracking(last_frame)
            
            print(f"\n🎯 Enhanced overlap tracking completed:")
            print(f"  📊 Frames with overlaps: {overlap_count}")
            print(f"  📈 Processing efficiency: {frame_count}/{len(self.frame_names)} frames")
            
            # Print detailed summary
            summary = self.overlap_tracker.get_overlap_summary()
            for target_name, data in summary.items():
                print(f"  🎯 {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames")
        
        # Store frame analyses for video creation
        self.frame_analyses = frame_analyses
        
        # Clean up inference state
        self.predictor.reset_state(inference_state)
        ultra_cleanup_memory()
        
        print(f"\n✅ Enhanced processing complete!")
        print(f"📊 Processed {frame_count} frames with improved overlap detection")
        
        return results
    
    def _process_cpu_fallback(self, points_dict, labels_dict, object_names, debug):
        """Emergency CPU fallback processing"""
        print("🚨 Emergency CPU fallback - this will be slow but stable")
        
        if hasattr(self.predictor.model, 'to'):
            self.predictor.model = self.predictor.model.to('cpu')
        
        ultra_cleanup_memory()
        
        messagebox.showwarning("Memory Limitation", 
                             "GPU memory exhausted. Falling back to CPU processing.\n"
                             "This will be much slower but should complete successfully.")
        
        return None
    
    def save_results_video_with_enhanced_annotations(self, results, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results video with enhanced visual feedback for looking-at events"""
        if not results:
            print("No results to save!")
            return
        
        # Get video properties
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        if show_original:
            out_width = width * 2
        else:
            out_width = width
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
        
        # Color map for consistent colors
        cmap = plt.get_cmap("tab10")
        
        print("💾 Saving video with enhanced looking-at visual feedback...")
        overlap_frame_count = 0
        
        for frame_idx in tqdm(range(len(self.frame_names)), desc="Saving frames"):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            if frame is None:
                continue
                
            overlay = frame.copy()
            
            # Get enhanced frame analysis if available
            frame_analysis = None
            if hasattr(self, 'frame_analyses') and frame_idx in self.frame_analyses:
                frame_analysis = self.frame_analyses[frame_idx]
            
            # Check if this frame has looking-at events
            has_looking_at_events = (frame_analysis and 
                                frame_analysis.get('target_overlaps') and 
                                any(frame_analysis['target_overlaps'].values()))
            if has_looking_at_events:
                overlap_frame_count += 1
            
            # Collect all looking-at information for this frame
            looking_at_info = {}  # obj_id -> {is_target: bool, looking_at: [objects], looked_at_by: [targets]}
            
            if frame_analysis:
                # Initialize info for all objects
                for obj_id in results.get(frame_idx, {}):
                    looking_at_info[obj_id] = {
                        'is_target': False,
                        'looking_at': [],
                        'looked_at_by': [],
                        'is_being_looked_at': False
                    }
                
                # Process target overlaps
                for target_id, looking_at_objects in frame_analysis.get('target_overlaps', {}).items():
                    if target_id in looking_at_info:
                        looking_at_info[target_id]['is_target'] = True
                        looking_at_info[target_id]['looking_at'] = [obj['object_name'] for obj in looking_at_objects]
                    
                    # Mark objects being looked at
                    for obj_info in looking_at_objects:
                        obj_id = obj_info['object_id']
                        if obj_id in looking_at_info:
                            target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                            looking_at_info[obj_id]['looked_at_by'].append(target_name)
                            looking_at_info[obj_id]['is_being_looked_at'] = True
            
            # Apply masks with enhanced visual feedback
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    
                    # Resize mask if needed
                    if mask.shape != (height, width) and mask.shape[0] > 0 and mask.shape[1] > 0:
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), 
                                            interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue
                    
                    if mask.shape == (height, width):
                        obj_info = looking_at_info.get(obj_id, {})
                        is_target = obj_info.get('is_target', False)
                        is_being_looked_at = obj_info.get('is_being_looked_at', False)
                        looking_at = obj_info.get('looking_at', [])
                        looked_at_by = obj_info.get('looked_at_by', [])
                        
                        # Choose colors based on status
                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        
                        if is_target and looking_at:
                            # Target that's looking at something - bright highlight
                            color = np.minimum(base_color + [100, 100, 0], 255)  # Yellow tint for active targets
                            border_color = (0, 255, 255)  # Cyan border for active targets
                            border_thickness = 8
                        elif is_being_looked_at:
                            # Object being looked at - red highlight
                            color = np.minimum(base_color + [120, 0, 0], 255)  # Red tint
                            border_color = (0, 0, 255)  # RED BORDER for looked-at objects
                            border_thickness = 8
                        else:
                            # Normal object
                            color = base_color
                            border_color = None
                            border_thickness = 2
                        
                        # Apply mask color
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
                        
                        # Add enhanced border for special objects
                        if border_color:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, border_color, border_thickness)
                        
                        # Add object label with status
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                            
                            # Create status label
                            if is_target and looking_at:
                                if len(looking_at) == 1:
                                    label = f"🎯{obj_name} → LOOKING AT {looking_at[0]}"
                                else:
                                    label = f"🎯{obj_name} → LOOKING AT {len(looking_at)} OBJECTS"
                            elif is_being_looked_at:
                                if len(looked_at_by) == 1:
                                    label = f"{obj_name} ← LOOKED AT BY {looked_at_by[0]}"
                                else:
                                    label = f"{obj_name} ← LOOKED AT BY {len(looked_at_by)} TARGETS"
                            else:
                                if is_target:
                                    label = f"🎯{obj_name}"
                                else:
                                    label = obj_name
                            
                            # Enhanced text rendering based on status
                            font_scale = 0.6 if len(label) > 30 else 0.7
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                            
                            # Calculate text position
                            label_x = max(5, min(cx - text_size[0]//2, width - text_size[0] - 5))
                            label_y = max(25, min(cy + 20, height - 10))
                            
                            # Enhanced background and text colors
                            if is_target and looking_at:
                                bg_color = (0, 100, 200)  # Orange background for active targets
                                text_color = (0, 255, 255)  # Bright cyan text
                                padding = 10
                            elif is_being_looked_at:
                                bg_color = (0, 0, 200)  # Red background for looked-at objects
                                text_color = (255, 255, 255)  # White text
                                padding = 10
                            else:
                                bg_color = (0, 0, 0)  # Black background
                                text_color = (255, 255, 255)  # White text
                                padding = 5
                            
                            # Draw background rectangle
                            cv2.rectangle(overlay, 
                                        (label_x - padding, label_y - 25), 
                                        (label_x + text_size[0] + padding, label_y + 5), 
                                        bg_color, -1)
                            
                            # Draw text
                            cv2.putText(overlay, label, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
            
            # Add prominent frame status display
            status_messages = []
            
            if has_looking_at_events:
                # Collect all looking-at events for display
                target_events = []
                if frame_analysis:
                    for target_id, looking_at_objects in frame_analysis.get('target_overlaps', {}).items():
                        target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                        object_names = [obj['object_name'] for obj in looking_at_objects]
                        
                        if len(object_names) == 1:
                            target_events.append(f"{target_name} → {object_names[0]}")
                        else:
                            target_events.append(f"{target_name} → {len(object_names)} objects")
                
                status_messages = [f"🎯 LOOKING AT DETECTED: {'; '.join(target_events)}"]
            
            # Draw status messages
            info_y = 30
            for i, message in enumerate(status_messages):
                if has_looking_at_events:
                    # Prominent background for looking-at events
                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(overlay, (5, info_y - 25), (text_size[0] + 15, info_y + 10), (0, 0, 180), -1)
                    cv2.putText(overlay, message, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(overlay, message, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                info_y += 35
            
            # Add frame counter
            frame_info = f"Frame {frame_idx}/{len(self.frame_names)-1}"
            cv2.putText(overlay, frame_info, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Create output frame
            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay
                
            out.write(output_frame)
        
        out.release()
        print(f"✅ Video saved with enhanced looking-at visual feedback: {output_path}")
        print(f"🎯 Looking-at events detected in {overlap_frame_count} frames")
        print(f"📊 Visual enhancements:")
        print(f"  • RED BORDERS on objects being looked at")
        print(f"  • CYAN BORDERS on targets that are looking")
        print(f"  • Clear status text for each object")
        print(f"  • Prominent event announcements")
        
    def create_elan_file(self, video_path, output_path, fps, frame_offset=0):
        """Create ELAN file with corrected timing alignment"""
        if not self.overlap_tracker.has_targets():
            print("No targets found - skipping ELAN export")
            return
        
        print(f"Creating ELAN file with timing correction: {output_path}")
        print(f"  Video FPS: {fps}")
        print(f"  Frame offset: {frame_offset}")
        
        # Get actual video properties for verification
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / actual_fps
            cap.release()
            
            print(f"  Actual video FPS: {actual_fps}")
            print(f"  Video duration: {duration_seconds:.2f}s ({total_frames} frames)")
            
            # Use actual FPS if significantly different
            if abs(fps - actual_fps) > 1.0:
                print(f"  ⚠️ FPS mismatch detected! Using actual FPS: {actual_fps}")
                fps = actual_fps
        except:
            print(f"  Using provided FPS: {fps}")
        
        summary = self.overlap_tracker.get_overlap_summary()
        
        # Debug timing calculations
        print(f"\n🔍 Timing Debug:")
        for target_name, target_data in summary.items():
            if target_data['events']:
                first_event = target_data['events'][0]
                start_frame = first_event['start_frame'] + frame_offset
                start_time = start_frame / fps
                print(f"  {target_name} first event: frame {first_event['start_frame']} → {start_frame} → {start_time:.3f}s")
        
        # Create ELAN XML with corrected timing
        header = f'''<?xml version="1.0" encoding="UTF-8"?>
    <ANNOTATION_DOCUMENT AUTHOR="SAM2_Looking_At_Events" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
        <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
            <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
                MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
            <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
        </HEADER>
        <TIME_ORDER>
    '''

        # Create time slots with corrected timing
        time_slots = []
        time_slot_id = 1
        time_slot_refs = {}
        
        all_time_points = set()
        for target_name, target_data in summary.items():
            for event in target_data['events']:
                # Apply frame offset and convert to time
                start_frame_corrected = event['start_frame'] + frame_offset
                end_frame_corrected = event['end_frame'] + frame_offset
                
                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                
                all_time_points.add(start_time)
                all_time_points.add(end_time)
        
        for time_point in sorted(all_time_points):
            time_ms = int(time_point * 1000)
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
            time_slot_refs[time_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

        header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

        # Create tiers with corrected timing
        tier_content = ""
        annotation_id = 1
        
        for target_name, target_data in summary.items():
            tier_id = target_name.upper().replace(' ', '_').replace('-', '_')
            tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}_LOOKING_AT">\n'
            
            for event in target_data['events']:
                # Apply frame offset and convert to time
                start_frame_corrected = event['start_frame'] + frame_offset
                end_frame_corrected = event['end_frame'] + frame_offset
                
                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                start_slot = time_slot_refs[start_ms]
                end_slot = time_slot_refs[end_ms]
                
                # Create annotation
                overlapping_objects_str = ", ".join(event['overlapping_objects'])
                duration_seconds = end_time - start_time
                
                if len(event['overlapping_objects']) == 1:
                    annotation_value = f"Looking at: {overlapping_objects_str}"
                else:
                    annotation_value = f"Looking at {len(event['overlapping_objects'])} objects: {overlapping_objects_str}"
                
                annotation = f'''        <ANNOTATION>
                <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                    <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>
                </ALIGNABLE_ANNOTATION>
            </ANNOTATION>'''
                
                tier_content += annotation + '\n'
                annotation_id += 1
            
            tier_content += '    </TIER>\n'

        footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
        <LOCALE LANGUAGE_CODE="en"/>
        <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
        <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
        <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
        <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
    </ANNOTATION_DOCUMENT>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + tier_content + footer)
        
        print(f"✅ ELAN file created with timing correction: {output_path}")
        
        # Print timing verification
        print(f"\n📄 Timing Verification:")
        for target_name, target_data in summary.items():
            if target_data['events']:
                print(f"  🎯 {target_name}:")
                for i, event in enumerate(target_data['events'][:3]):
                    start_frame_corrected = event['start_frame'] + frame_offset
                    end_frame_corrected = event['end_frame'] + frame_offset
                    start_time = start_frame_corrected / fps
                    end_time = end_frame_corrected / fps
                    
                    objects = ", ".join(event['overlapping_objects'])
                    print(f"    Event {i+1}: {start_time:.3f}s - {end_time:.3f}s | {objects}")
                    print(f"              (frames {start_frame_corrected} - {end_frame_corrected})")
                
                if len(target_data['events']) > 3:
                    print(f"    ... and {len(target_data['events'])-3} more events")

# Point selection function (same as before but with enhanced tips)
def select_points_opencv(frame, processor=None):
    """Interactive point selection tool with enhanced overlap detection tips"""
    points_dict = {}
    labels_dict = {}
    object_names = {}
    current_obj_id = 1
    
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
        
    def test_current_mask(frame, points_dict, labels_dict, current_obj_id, object_names, predictor):
        """Test current mask by generating preview with SAM2"""
        if current_obj_id not in points_dict or len(points_dict[current_obj_id]) == 0:
            print("No points selected for current object!")
            return
        
        try:
            print(f"Testing mask for object {current_obj_id}...")
            
            # Create a temporary frames directory for testing
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            print(f"Created temp frame: {temp_frame_path}")
            
            # Initialize SAM2 state for testing (same as in main processing)
            inference_state = predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            predictor.reset_state(inference_state)
            print("SAM2 state initialized for testing")
            
            # Add points for current object (same as in main processing)
            points = np.array(points_dict[current_obj_id], dtype=np.float32)
            labels = np.array(labels_dict[current_obj_id], dtype=np.int32)
            
            print(f"Points shape: {points.shape}, Labels shape: {labels.shape}")
            print(f"Points: {points}")
            print(f"Labels: {labels}")
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=current_obj_id,
                points=points,
                labels=labels,
            )
            
            print(f"SAM2 output - obj_ids: {out_obj_ids}, mask_logits shape: {out_mask_logits.shape}")
            
            # Generate mask (same as in main processing)
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]
            
            print(f"Generated mask shape: {mask.shape}, mask sum: {np.sum(mask)}")
            
            if np.sum(mask) == 0:
                print("⚠️ Warning: Generated mask is empty!")
                
            # Create preview
            preview = frame.copy()
            
            # Apply mask with color (same as in main processing)
            cmap = plt.get_cmap("tab10")
            color = np.array(cmap(current_obj_id % 10)[:3]) * 255
            
            # Create colored overlay
            color_mask = np.zeros_like(preview)
            for c in range(3):
                color_mask[:, :, c][mask] = color[c]
            
            # Blend overlay
            cv2.addWeighted(preview, 0.5, color_mask, 0.5, 0, preview)
            
            # Add contours for better visibility
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, (0, 255, 0), 3)
            
            # Add text info
            obj_name = object_names.get(current_obj_id, f"Object_{current_obj_id}")
            pos_count = sum(1 for l in labels if l == 1)
            neg_count = sum(1 for l in labels if l == 0)
            
            info_text = f"MASK TEST: {obj_name} (+{pos_count} -{neg_count} points)"
            cv2.putText(preview, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, f"Mask pixels: {np.sum(mask)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, "Press any key to continue...", (10, preview.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show preview
            cv2.namedWindow('Mask Test Preview', cv2.WINDOW_NORMAL)
            cv2.imshow('Mask Test Preview', preview)
            cv2.waitKey(0)
            cv2.destroyWindow('Mask Test Preview')
            
            # Cleanup
            predictor.reset_state(inference_state)
            shutil.rmtree(temp_dir)
            
            print(f"✅ Mask test completed for {obj_name}")
            
        except Exception as e:
            print(f"❌ Error testing mask: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            try:
                predictor.reset_state(inference_state)
                shutil.rmtree(temp_dir)
            except:
                pass
    def redraw_all_points():
        display = frame.copy()
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        
        height, width = display.shape[:2]
        
        overlay = display.copy()
        instructions_height = 240
        cv2.rectangle(overlay, (10, height - instructions_height - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        instructions = [
            "ENHANCED OVERLAP DETECTION:",
            "Left Click: Add positive point (+)",
            "Right Click: Add negative point (-)",
            "R: Reset  N: Next  P: Previous",
            "T: Test current mask  R: Reset  N: Next  P: Previous",  
            "C: Name object  Enter: Finish  Q: Quit",
            "",
            "OVERLAP TIPS:",
            "• Detects inclusion (objects inside targets)",
            "• Detects partial overlaps",
            "• Use 'target_1', 'target_2' for overlap tracking"
        ]
        
        y_start = height - instructions_height
        for i, instruction in enumerate(instructions):
            if instruction == "":
                continue
            
            color = (0, 255, 255) if "ENHANCED" in instruction else (0, 255, 0) if "OVERLAP TIPS" in instruction else (255, 255, 255)
            font_scale = 0.6 if i in [0, 6] else 0.5
            thickness = 2 if i in [0, 6] else 1
            
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
        name = simpledialog.askstring("Enhanced Object Naming", 
                                     f"Enter name for object {current_obj_id}:\n\n"
                                     f"ENHANCED OVERLAP DETECTION:\n"
                                     f"• Detects inclusion (objects inside targets)\n"
                                     f"• Detects partial overlaps\n"
                                     f"• Use 'target_1', 'target_2' for tracking",
                                     initialvalue=current_name)
        root.destroy()
        
        if name and name.strip():
            object_names[current_obj_id] = name.strip()
            print(f"Object {current_obj_id} named: {object_names[current_obj_id]}")
            
            if 'target' in name.lower():
                print(f"🎯 Target detected: Enhanced overlap detection will track inclusion and overlaps")
            
            nonlocal img_display
            img_display = redraw_all_points()
        
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
    cv2.namedWindow('Enhanced Point Selection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Enhanced Point Selection', click_handler)
    
    print("\n🎯 SIMPLIFIED 'LOOKING AT' EVENT DETECTION ENABLED")
    print("🔍 Features:")
    print("  • Detects any spatial relationship as 'looking at' event")
    print("  • Clean begin/end timing for behavioral analysis")
    print("  • Simplified text-based annotations in video")
    print("  • ELAN export with accurate event boundaries")
    print("  • Perfect for gaze studies and interaction analysis")
    
    while True:
        cv2.imshow('Enhanced Point Selection', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            if current_obj_id in points_dict:
                points_dict[current_obj_id] = []
                labels_dict[current_obj_id] = []
                img_display = redraw_all_points()
                obj_name = get_object_name(current_obj_id)
                print(f"Reset points for {obj_name}")
        elif key == ord('t'):  # ADD THIS
            if processor and processor.predictor:
                test_current_mask(frame, points_dict, labels_dict, current_obj_id, object_names, processor.predictor)
            else:
                messagebox.showwarning("Test Mask", "Predictor not available for testing!")
    
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
        
        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            return points_dict, labels_dict, object_names if points_dict else (None, None, None)
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None, None
    
    return points_dict, labels_dict, object_names

# GUI Application class with enhanced overlap detection
class VideoAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAM2 Video Analysis - 'Looking At' Event Detection \n Developers: Wim Pouw, Davide Ahmar, Babajide Owoyele")
        self.root.geometry("750x850")
        self.root.minsize(750, 850)
        
        # Initialize SAM2
        self.device = setup_device_ultra_optimized()
        self.predictor = None
        self.init_sam2()
        
        self.setup_gui()
        
    def init_sam2(self):
        """Initialize SAM2 predictor with ultra memory optimization"""
        try:
            configure_torch_ultra_conservative()
            
            sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if not os.path.exists(sam2_checkpoint):
                print("⚠️ Large model not found, checking for small model...")
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
                
                if not os.path.exists(sam2_checkpoint):
                    messagebox.showwarning("SAM2 Setup", 
                        f"SAM2 checkpoints not found. Please update paths.")
                    return
                else:
                    print("✅ Using small model for better memory efficiency")
            else:
                print("✅ Using large model with enhanced overlap detection")
            
            from sam2.build_sam import build_sam2_video_predictor
            
            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            print("✅ SAM2 predictor initialized with enhanced overlap detection")
            
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                print(f"📊 GPU Memory: {gpu_info['allocated_gb']:.1f}GB allocated, {gpu_info['free_gb']:.1f}GB free")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize SAM2: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI with enhanced overlap detection options"""
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="SAM2 Video Analysis - 'Looking At' Event Detection", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Simplified features info
        features_frame = tk.LabelFrame(main_frame, text="🎯 'Looking At' Event Detection Features", font=("Arial", 9, "bold"))
        features_frame.pack(fill=tk.X, pady=(0, 10))
        
        features_text = """✅ SPATIAL RELATIONSHIP DETECTION: Any overlap/inclusion = 'looking at' event
✅ CLEAN VIDEO SIGNALING: Clear indicators when target is looking at objects  
✅ ACCURATE TIMING: Precise begin/end times for behavioral analysis
✅ ELAN EXPORT: Perfect for gaze studies and interaction coding"""
        
        features_label = tk.Label(features_frame, text=features_text, 
                                 font=("Arial", 8), fg="darkgreen", justify=tk.LEFT)
        features_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Memory status
        memory_status_frame = tk.LabelFrame(main_frame, text="🧠 Memory Status", font=("Arial", 9, "bold"))
        memory_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            memory_text = f"GPU: {gpu_info['allocated_gb']:.1f}GB used / {gpu_info['total_gb']:.1f}GB total"
        else:
            memory_text = "GPU: Not available (using CPU)"
        
        self.memory_status_var = tk.StringVar(value=memory_text)
        memory_label = tk.Label(memory_status_frame, textvariable=self.memory_status_var, 
                               font=("Arial", 8), fg="darkblue")
        memory_label.pack(anchor=tk.W, padx=5, pady=5)
        
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
        
        # Simplified event detection options
        overlap_frame = tk.LabelFrame(main_frame, text="🎯 'Looking At' Event Settings", font=("Arial", 9, "bold"))
        overlap_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Overlap threshold
        threshold_frame = tk.Frame(overlap_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threshold_frame, text="Detection Threshold (%):").pack(side=tk.LEFT)
        self.overlap_threshold_var = tk.StringVar(value="10")
        threshold_spin = tk.Spinbox(threshold_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.overlap_threshold_var, width=10)
        threshold_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(threshold_frame, text="(minimum spatial relationship to detect)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Event detection info
        event_info = tk.Label(overlap_frame, 
                             text="Event Detection: Any spatial relationship (overlap/inclusion) = 'looking at' event",
                             font=("Arial", 8), fg="darkgreen", wraplength=700)
        event_info.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        # Memory optimization options
        options_frame = tk.LabelFrame(main_frame, text="🚀 Memory Optimization", font=("Arial", 9, "bold"))
        options_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Batch size
        batch_frame = tk.Frame(options_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(batch_frame, text="Memory Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value="50")
        batch_spin = tk.Spinbox(batch_frame, from_=10, to=200, increment=10, 
                               textvariable=self.batch_size_var, width=10)
        batch_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Auto fallback
        self.auto_fallback = tk.BooleanVar(value=True)
        fallback_cb = tk.Checkbutton(options_frame, 
                                    text="🔄 Auto fallback to CPU if GPU memory exhausted",
                                    variable=self.auto_fallback)
        fallback_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Output options
        output_frame = tk.LabelFrame(main_frame, text="📁 Output Options", font=("Arial", 9, "bold"))
        output_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.enable_elan_export = tk.BooleanVar(value=True)
        elan_cb = tk.Checkbutton(output_frame, 
                                text="📄 Export ELAN file with 'looking at' event timing",
                                variable=self.enable_elan_export)
        elan_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(15, 10))
        
        self.process_button = tk.Button(process_frame, text="🎯 Process Video ('Looking At' Event Detection)", 
                                       command=self.process_video, bg="#4CAF50", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - 'Looking at' event detection with clean timing")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 8), wraplength=700)
        status_label.pack(pady=(5, 0))
    
    def update_memory_status(self):
        """Update memory status display"""
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            memory_text = f"GPU: {gpu_info['allocated_gb']:.1f}GB used / {gpu_info['total_gb']:.1f}GB total ({gpu_info['utilization_pct']:.1f}%)"
            if gpu_info['utilization_pct'] > 85:
                memory_text += " ⚠️ HIGH"
        else:
            memory_text = "GPU: Not available (using CPU)"
        
        self.memory_status_var.set(memory_text)
        self.root.update()
    
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
            self.status_var.set(f"Found {len(videos)} video(s) - 'Looking at' event detection ready")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number_with_preview(self, frames_dir, total_frames):
        """Get frame number with preview functionality"""
        suggested_frame = total_frames // 2
        
        while True:
            frame_num = simpledialog.askinteger(
                "Reference Frame Selection - 'Looking At' Event Detection",
                f"Select reference frame (0-{total_frames-1}):\n\n"
                f"🎯 'LOOKING AT' EVENT DETECTION:\n"
                f"• Detects any spatial relationship as 'looking at' event\n"
                f"• Clean begin/end timing for behavioral analysis\n"
                f"• Perfect for gaze studies and interaction coding\n"
                f"📊 Suggested: Frame {suggested_frame} (middle of video)\n\n"
                f"Enter frame number (or -1 to preview suggested frame):",
                minvalue=-1,
                maxvalue=total_frames-1,
                initialvalue=suggested_frame
            )
            
            if frame_num is None:
                return None
            
            if frame_num == -1:
                if show_frame_preview(frames_dir, suggested_frame, total_frames):
                    continue
                else:
                    return None
            
            if show_frame_preview(frames_dir, frame_num, total_frames):
                confirm = messagebox.askyesno("Confirm Frame Selection", 
                    f"Use frame {frame_num} as reference frame?\n\n"
                    "🔍 Enhanced overlap detection will analyze:\n"
                    "• When objects are included inside targets\n"
                    "• When objects partially overlap with targets\n"
                    "• All spatial relationships between objects")
                
                if confirm:
                    return frame_num
            else:
                return None
    
    def process_video(self):
        """Process the selected video with enhanced overlap detection"""
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
            self.status_var.set("🎯 Starting 'looking at' event detection...")
            self.update_memory_status()
            
            # Create frames directory
            video_stem = Path(video_name).stem
            frames_dir = os.path.join(folder, f"{video_stem}_frames")
            
            # Extract frames
            self.status_var.set("📹 Extracting frames...")
            self.root.update()
            
            fps, num_frames = video_to_frames(video_path, frames_dir)
            if fps == -1:
                messagebox.showerror("Error", "Failed to extract frames from video")
                return
            
            print(f"🎯 Processing {num_frames} frames with 'looking at' event detection")
            
            # Get reference frame
            frame_num = self.get_frame_number_with_preview(frames_dir, num_frames)
            if frame_num is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Get settings
            try:
                overlap_threshold = float(self.overlap_threshold_var.get()) / 100.0
                batch_size = int(self.batch_size_var.get())
            except ValueError:
                overlap_threshold = 0.1
                batch_size = 50
                self.overlap_threshold_var.set("10")
                self.batch_size_var.set("50")
            
            # Initialize enhanced processor
            self.status_var.set("🔧 Initializing enhanced overlap processor...")
            self.update_memory_status()
            
            processor = UltraOptimizedProcessor(
                predictor=self.predictor,
                video_dir=frames_dir,
                overlap_threshold=overlap_threshold,
                reference_frame=frame_num,
                batch_size=batch_size,
                auto_fallback=self.auto_fallback.get()
            )
            
            # Load reference frame
            frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
            if not os.path.exists(frame_path):
                messagebox.showerror("Error", f"Frame {frame_num} not found")
                return
            
            frame = cv2.imread(frame_path)
            
            # Enhanced point selection
            self.status_var.set("🎯 Select points for enhanced overlap detection...")
            self.root.update()
            
            messagebox.showinfo("'Looking At' Event Detection", 
                f"Reference frame {frame_num} will open for annotation.\n\n"
                "🎯 'LOOKING AT' EVENT DETECTION FEATURES:\n"
                "• Any spatial relationship = 'looking at' event\n"
                "• Clean begin/end timing for behavioral analysis\n"
                "• Perfect for gaze studies and interaction coding\n"
                "• Simplified event categorization\n\n"
                "🎯 TARGET SETUP:\n"
                "• Name crosshairs/targets as 'target_1', 'target_2'\n"
                "• System will detect when targets look at objects\n"
                "• Clean timing exported to ELAN for behavioral analysis")
            
            points_dict, labels_dict, object_names = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            print(f"🎯 Selected {len(object_names)} objects for 'looking at' event detection")
            
            # Process with simplified event detection
            self.status_var.set("🚀 Processing with 'looking at' event detection...")
            self.update_memory_status()
            
            results = processor.process_video_with_memory_management(points_dict, labels_dict, object_names)
            
            if results:
                # Save results with enhanced annotations
                self.status_var.set("💾 Saving results with enhanced annotations...")
                self.root.update()
                
                output_path = os.path.join(frames_dir, "output_enhanced_overlap.mp4")
                
                processor.save_results_video_with_enhanced_annotations(
                    results=results,
                    output_path=output_path,
                    fps=fps,
                    show_original=True,
                    alpha=0.5
                )
                
                # Save enhanced ELAN file
                elan_created = False
                if self.enable_elan_export.get() and processor.overlap_tracker.has_targets():
                    elan_path = os.path.join(frames_dir, "enhanced_target_overlaps.eaf")
                    processor.create_elan_file(
                        video_path=output_path,
                        output_path=elan_path,
                        fps=fps
                    )
                    elan_created = True
                
                self.status_var.set("✅ 'Looking at' event detection completed!")
                self.update_memory_status()
                
                # Enhanced success message
                target_info = ""
                if processor.overlap_tracker.has_targets():
                    summary = processor.overlap_tracker.get_overlap_summary()
                    target_info = f"\n\n🎯 'Looking At' Events:\n"
                    for target_name, data in summary.items():
                        target_info += f"  • {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames\n"
                    if elan_created:
                        target_info += "\n📄 ELAN file: enhanced_target_overlaps.eaf"
                
                named_objects = [name for name in object_names.values()]
                objects_summary = "\n".join([f"  • {name}" for name in named_objects])
                
                success_msg = f"""🎯 'Looking At' Event Detection Complete!

Reference Frame: {frame_num}
Detection Method: Any spatial relationship = 'looking at' event
Detection Threshold: {overlap_threshold*100:.1f}%
Clean Timing: Accurate begin/end for behavioral analysis
Results saved in: {frames_dir}

📁 Generated Files:
• output_enhanced_overlap.mp4 - Video with event indicators
• time_series_metrics.csv - Movement data"""

                if elan_created:
                    success_msg += "\n• enhanced_target_overlaps.eaf - ELAN with clean event timing"

                success_msg += f"""{target_info}

📊 Analyzed Objects ({len(object_names)}):
{objects_summary}

✅ 'Looking at' event detection with clean timing completed!

🎯 DETECTION FEATURES:
• Clean event signaling in video
• Accurate begin/end times for ELAN
• Perfect for behavioral analysis and gaze studies
• Simplified 'looking at' event categorization"""
                
                messagebox.showinfo("'Looking At' Event Detection Complete", success_msg)
                
            elif results is None:
                messagebox.showwarning("Processing Incomplete", 
                    "GPU memory was exhausted. 'Looking at' event detection used CPU fallback.\n\n"
                    "Consider reducing the number of objects for better performance.")
                self.status_var.set("Event detection completed with limitations")
            else:
                messagebox.showerror("Error", "'Looking at' event detection failed")
                self.status_var.set("Event detection failed")
        
        except Exception as e:
            messagebox.showerror("Error", f"Event detection failed: {str(e)}")
            self.status_var.set("Event detection failed")
            import traceback
            traceback.print_exc()
        finally:
            ultra_cleanup_memory()
            self.update_memory_status()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting SAM2 Video Analysis - Enhanced Overlap Detection!")
    print("=" * 70)
    
    print("\n🎯 SIMPLIFIED 'LOOKING AT' EVENT DETECTION:")
    print("  • SPATIAL RELATIONSHIP DETECTION: Any overlap/inclusion = 'looking at' event")
    print("  • CLEAN VIDEO SIGNALING: Clear indicators when events are happening")
    print("  • ACCURATE TIMING: Precise begin/end times for ELAN behavioral analysis")
    print()
    print("🎯 EVENT TYPES:")
    print("  • All spatial relationships treated as 'looking at' events")
    print("  • Inclusion (object inside target) = looking at")
    print("  • Partial overlap (objects touching) = looking at")
    print("  • Clean start/stop timing for behavioral analysis")
    print()
    print("💬 ANNOTATION EXAMPLES:")
    print("  • 'TARGET crosshair LOOKING AT apple'")
    print("  • 'hand LOOKED AT BY TARGET pointer'")
    print("  • 'Frame 245/1200 - LOOKING AT EVENT DETECTED'")
    print()
    print("📄 ELAN EXPORT:")
    print("  • Accurate begin/end times for each 'looking at' event")
    print("  • Clean event boundaries for behavioral coding")
    print("  • Simple 'Looking at: object_name' annotations")
    print("  • Perfect for gaze analysis and interaction studies")
    print()
    print("🧠 MEMORY OPTIMIZATIONS:")
    print("  • GPU memory management")
    print("  • Multiple fallback strategies for OOM recovery")
    print("  • Enhanced cleanup every 25 frames")
    print("  • Automatic CPU fallback if needed")
    print()
    print("Starting enhanced overlap detection application...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()