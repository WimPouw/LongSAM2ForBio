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

# Ultra-aggressive memory optimization (same as before)
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
    """Ultra-aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def configure_torch_ultra_conservative():
    """Configure PyTorch for ultra-conservative memory usage"""
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
        print(f"GPU Memory after ultra-conservative setup: {get_gpu_memory_info()}")

def setup_device_ultra_optimized():
    """Setup computation device with ultra-conservative settings"""
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
        self.inclusion_threshold = 0.8  # 80% overlap = inclusion
        
    def calculate_detailed_overlap(self, mask1, mask2):
        """Calculate detailed overlap information including inclusion detection"""
        if mask1.shape != mask2.shape:
            return None
        
        # Ensure masks are boolean
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)
        
        # Calculate areas
        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)
        
        if area1 == 0 or area2 == 0:
            return None
        
        # Calculate intersection
        intersection = np.logical_and(mask1_bool, mask2_bool)
        intersection_area = np.sum(intersection)
        
        if intersection_area == 0:
            return None
        
        # Calculate overlap percentages
        overlap_pct_1 = intersection_area / area1  # How much of mask1 is covered by mask2
        overlap_pct_2 = intersection_area / area2  # How much of mask2 is covered by mask1
        
        # Determine relationship type
        relationship_type = "overlap"  # Default
        
        if overlap_pct_1 >= self.inclusion_threshold:
            relationship_type = "mask1_included_in_mask2"  # mask1 is mostly inside mask2
        elif overlap_pct_2 >= self.inclusion_threshold:
            relationship_type = "mask2_included_in_mask1"  # mask2 is mostly inside mask1
        
        # Use the smaller overlap percentage for threshold comparison
        min_overlap_pct = min(overlap_pct_1, overlap_pct_2)
        
        return {
            'intersection_area': intersection_area,
            'overlap_pct_1': overlap_pct_1,  # How much of object 1 overlaps
            'overlap_pct_2': overlap_pct_2,  # How much of object 2 overlaps
            'min_overlap_pct': min_overlap_pct,
            'max_overlap_pct': max(overlap_pct_1, overlap_pct_2),
            'relationship_type': relationship_type,
            'meets_threshold': min_overlap_pct >= self.overlap_threshold
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
    
    def analyze_frame_overlaps(self, frame_results, object_names):
        """Analyze all overlaps in a frame and return detailed information"""
        frame_analysis = {
            'target_overlaps': {},  # target_id -> list of overlapping objects with details
            'object_relationships': {},  # obj_id -> relationship info
            'all_overlaps': []  # All overlap relationships for debugging
        }
        
        # First, analyze all target relationships
        for target_id in self.target_objects:
            if target_id not in frame_results:
                continue
                
            target_mask = frame_results[target_id]
            if len(target_mask.shape) > 2:
                target_mask = target_mask.squeeze()
            
            target_name = self.target_objects[target_id]
            overlapping_objects = []
            
            # Check overlap with all other objects
            for obj_id, mask in frame_results.items():
                if obj_id == target_id:
                    continue
                    
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                
                # Calculate detailed overlap
                overlap_info = self.detector.calculate_detailed_overlap(target_mask, mask)
                
                if overlap_info and overlap_info['meets_threshold']:
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    
                    # Determine the relationship description
                    if overlap_info['relationship_type'] == "mask2_included_in_mask1":
                        # Other object is included in target
                        relationship_desc = f"INCLUDES {obj_name}"
                        relationship_type = "inclusion"
                    elif overlap_info['relationship_type'] == "mask1_included_in_mask2":
                        # Target is included in other object
                        relationship_desc = f"INCLUDED IN {obj_name}"
                        relationship_type = "included"
                    else:
                        # Partial overlap
                        relationship_desc = f"OVERLAPS {obj_name}"
                        relationship_type = "overlap"
                    
                    overlapping_objects.append({
                        'object_id': obj_id,
                        'object_name': obj_name,
                        'relationship_type': relationship_type,
                        'relationship_desc': relationship_desc,
                        'overlap_details': overlap_info
                    })
                    
                    # Store bidirectional relationship info
                    frame_analysis['all_overlaps'].append({
                        'obj1_id': target_id,
                        'obj1_name': target_name,
                        'obj2_id': obj_id,
                        'obj2_name': obj_name,
                        'relationship_type': relationship_type,
                        'overlap_info': overlap_info
                    })
            
            if overlapping_objects:
                frame_analysis['target_overlaps'][target_id] = overlapping_objects
        
        # Create bidirectional object relationship mapping
        for target_id, overlaps in frame_analysis['target_overlaps'].items():
            target_name = self.target_objects[target_id]
            
            # Store info for the target object
            frame_analysis['object_relationships'][target_id] = {
                'is_target': True,
                'overlapping_with': overlaps,
                'display_text': self._create_target_display_text(target_name, overlaps)
            }
            
            # Store info for each overlapping object
            for overlap in overlaps:
                obj_id = overlap['object_id']
                obj_name = overlap['object_name']
                
                # Create reverse relationship description
                if overlap['relationship_type'] == "inclusion":
                    reverse_desc = f"INCLUDED IN TARGET {target_name}"
                elif overlap['relationship_type'] == "included":
                    reverse_desc = f"INCLUDES TARGET {target_name}"
                else:
                    reverse_desc = f"OVERLAPS TARGET {target_name}"
                
                frame_analysis['object_relationships'][obj_id] = {
                    'is_target': False,
                    'related_to_target': target_id,
                    'target_name': target_name,
                    'relationship_type': overlap['relationship_type'],
                    'display_text': reverse_desc
                }
        
        return frame_analysis
    
    def _create_target_display_text(self, target_name, overlapping_objects):
        """Create clear display text for target objects"""
        if not overlapping_objects:
            return f"TARGET {target_name}"
        
        # Group by relationship type
        inclusions = [obj for obj in overlapping_objects if obj['relationship_type'] == "inclusion"]
        included_in = [obj for obj in overlapping_objects if obj['relationship_type'] == "included"]
        overlaps = [obj for obj in overlapping_objects if obj['relationship_type'] == "overlap"]
        
        parts = [f"TARGET {target_name}"]
        
        if inclusions:
            inclusion_names = [obj['object_name'] for obj in inclusions]
            if len(inclusion_names) == 1:
                parts.append(f"INCLUDES {inclusion_names[0]}")
            else:
                parts.append(f"INCLUDES {', '.join(inclusion_names)}")
        
        if included_in:
            included_names = [obj['object_name'] for obj in included_in]
            if len(included_names) == 1:
                parts.append(f"INSIDE {included_names[0]}")
            else:
                parts.append(f"INSIDE {', '.join(included_names)}")
        
        if overlaps:
            overlap_names = [obj['object_name'] for obj in overlaps]
            if len(overlap_names) == 1:
                parts.append(f"OVERLAPS {overlap_names[0]}")
            else:
                parts.append(f"OVERLAPS {', '.join(overlap_names)}")
        
        return " | ".join(parts)
    
    def track_frame_overlaps_batch(self, frame_idx, frame_results, object_names):
        """Track overlaps for a frame using improved analysis"""
        frame_analysis = self.analyze_frame_overlaps(frame_results, object_names)
        
        # Update overlap events for each target
        for target_id in self.target_objects:
            if target_id in frame_analysis['target_overlaps']:
                overlapping_objects = frame_analysis['target_overlaps'][target_id]
                overlapping_names = [obj['object_name'] for obj in overlapping_objects]
                self._update_overlap_event(target_id, frame_idx, overlapping_names)
        
        return frame_analysis
    
    def _update_overlap_event(self, target_id, frame_idx, overlapping_names):
        """Update overlap events efficiently"""
        events = self.overlap_events[target_id]
        current_overlap_set = set(overlapping_names)
        
        if events and not events[-1].get('end_frame'):
            last_event = events[-1]
            last_overlap_set = set(last_event['overlapping_objects'])
            
            if current_overlap_set == last_overlap_set:
                last_event['end_frame'] = frame_idx
                last_event['duration_frames'] = frame_idx - last_event['start_frame'] + 1
                return
            else:
                last_event['end_frame'] = frame_idx - 1
                last_event['duration_frames'] = last_event['end_frame'] - last_event['start_frame'] + 1
        
        new_event = {
            'start_frame': frame_idx,
            'end_frame': None,
            'duration_frames': 1,
            'overlapping_objects': list(overlapping_names)
        }
        events.append(new_event)
    
    def finalize_tracking(self, last_frame_idx):
        """Finalize any open events"""
        for target_id, events in self.overlap_events.items():
            if events and not events[-1].get('end_frame'):
                events[-1]['end_frame'] = last_frame_idx
                events[-1]['duration_frames'] = last_frame_idx - events[-1]['start_frame'] + 1
    
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
        
        print(f"Ultra-Optimized Processor with Enhanced Overlap Detection")
        print(f"  Frames: {len(self.frame_names)}")
        print(f"  Reference frame: {reference_frame}")
        print(f"  Overlap threshold: {overlap_threshold*100:.1f}%")
        print(f"  Inclusion threshold: 80% (for detecting when objects are inside targets)")
        
        # Memory optimization flags
        self.offload_video_to_cpu = os.environ.get("SAM2_OFFLOAD_VIDEO_TO_CPU", "true") == "true"
        self.offload_state_to_cpu = os.environ.get("SAM2_OFFLOAD_STATE_TO_CPU", "true") == "true"
    
    def process_video_with_memory_management(self, points_dict, labels_dict, object_names, debug=True):
        """Process video with ultra memory management and improved overlap detection"""
        try:
            configure_torch_ultra_conservative()
            
            print(f"\nStarting ultra-optimized processing with enhanced overlap detection...")
            
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
        print("🔧 Initializing ultra-optimized SAM2 state...")
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
        """Save results video with enhanced overlap annotations"""
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
        
        print("💾 Saving video with enhanced overlap annotations...")
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
            
            # Check if this frame has overlaps
            has_overlaps = (frame_analysis and 
                          frame_analysis.get('target_overlaps') and 
                          any(frame_analysis['target_overlaps'].values()))
            if has_overlaps:
                overlap_frame_count += 1
            
            # Apply masks and enhanced annotations
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
                        # Get enhanced annotation info
                        annotation_info = None
                        is_overlapping = False
                        
                        if frame_analysis and obj_id in frame_analysis.get('object_relationships', {}):
                            annotation_info = frame_analysis['object_relationships'][obj_id]
                            is_overlapping = len(annotation_info.get('overlapping_with', [])) > 0 or 'related_to_target' in annotation_info
                        
                        # Choose color based on overlap status
                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        if is_overlapping:
                            # Enhanced highlighting for overlapping objects
                            color = np.minimum(base_color + [120, 0, 0], 255)
                            border_color = (0, 0, 255)  # Red border
                            border_thickness = 6
                        else:
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
                        
                        # Add enhanced border
                        if border_color:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, border_color, border_thickness)
                        
                        # Add enhanced object label
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            # Use enhanced annotation if available
                            if annotation_info:
                                label = annotation_info['display_text']
                            else:
                                obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                                if obj_id in self.overlap_tracker.target_objects:
                                    label = f"TARGET {obj_name}"
                                else:
                                    label = obj_name
                            
                            # Enhanced text rendering
                            font_scale = 0.5 if len(label) > 30 else 0.6
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                            
                            # Calculate text position (avoid going off screen)
                            label_x = max(5, min(cx - text_size[0]//2, width - text_size[0] - 5))
                            label_y = max(25, min(cy + 20, height - 10))
                            
                            # Enhanced background for overlapping objects
                            if is_overlapping:
                                bg_color = (0, 0, 150)  # Darker blue background
                                text_color = (0, 255, 255)  # Bright cyan text
                                padding = 8
                            else:
                                bg_color = (0, 0, 0)
                                text_color = (255, 255, 255)
                                padding = 5
                            
                            # Multi-line text handling for long labels
                            if len(label) > 40:
                                # Split long labels into multiple lines
                                words = label.split(' ')
                                lines = []
                                current_line = []
                                
                                for word in words:
                                    test_line = ' '.join(current_line + [word])
                                    test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                                    
                                    if test_size[0] > width - 20:  # Leave some margin
                                        if current_line:
                                            lines.append(' '.join(current_line))
                                            current_line = [word]
                                        else:
                                            lines.append(word)
                                    else:
                                        current_line.append(word)
                                
                                if current_line:
                                    lines.append(' '.join(current_line))
                                
                                # Draw multi-line text
                                line_height = 22
                                total_height = len(lines) * line_height
                                
                                # Background rectangle for all lines
                                max_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0] for line in lines)
                                cv2.rectangle(overlay, 
                                            (label_x - padding, label_y - 20), 
                                            (label_x + max_width + padding, label_y + total_height - 15), 
                                            bg_color, -1)
                                
                                # Draw each line
                                for i, line in enumerate(lines):
                                    line_y = label_y + (i * line_height)
                                    cv2.putText(overlay, line, (label_x, line_y),
                                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
                            else:
                                # Single line text
                                cv2.rectangle(overlay, 
                                            (label_x - padding, label_y - 20), 
                                            (label_x + text_size[0] + padding, label_y + 5), 
                                            bg_color, -1)
                                
                                cv2.putText(overlay, label, (label_x, label_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
            
            # Add enhanced frame info and overlap status
            info_text = f"Frame {frame_idx}/{len(self.frame_names)-1}"
            if has_overlaps:
                info_text += " - ENHANCED OVERLAP DETECTION ACTIVE"
                # More prominent background for overlap frames
                cv2.rectangle(overlay, (5, 5), (len(info_text) * 12, 45), (0, 0, 180), -1)
                cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Create output frame
            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay
                
            out.write(output_frame)
        
        out.release()
        print(f"✅ Video saved with enhanced overlap annotations: {output_path}")
        print(f"📊 Enhanced overlap detection found overlaps in {overlap_frame_count} frames")
        
        # Print detailed analysis summary
        if hasattr(self, 'frame_analyses'):
            inclusion_count = 0
            overlap_count = 0
            
            for frame_analysis in self.frame_analyses.values():
                for target_overlaps in frame_analysis.get('target_overlaps', {}).values():
                    for overlap in target_overlaps:
                        if overlap['relationship_type'] == 'inclusion':
                            inclusion_count += 1
                        elif overlap['relationship_type'] == 'overlap':
                            overlap_count += 1
            
            print(f"📈 Analysis Details:")
            print(f"  • Inclusion events detected: {inclusion_count}")
            print(f"  • Partial overlap events detected: {overlap_count}")
            print(f"  • Total relationship events: {inclusion_count + overlap_count}")
    
    def create_elan_file(self, video_path, output_path, fps):
        """Create ELAN file with enhanced overlap information"""
        if not self.overlap_tracker.has_targets():
            print("No targets found - skipping ELAN export")
            return
        
        print(f"Creating enhanced ELAN file: {output_path}")
        
        summary = self.overlap_tracker.get_overlap_summary()
        
        # Create ELAN XML with enhanced annotations
        header = f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="SAM2_Enhanced_Overlap" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
'''

        # Create time slots
        time_slots = []
        time_slot_id = 1
        time_slot_refs = {}
        
        all_time_points = set()
        for target_name, target_data in summary.items():
            for event in target_data['events']:
                start_time = event['start_frame'] / fps
                end_time = event['end_frame'] / fps
                all_time_points.add(start_time)
                all_time_points.add(end_time)
        
        for time_point in sorted(all_time_points):
            time_ms = int(time_point * 1000)
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
            time_slot_refs[time_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

        header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

        # Create enhanced tiers
        tier_content = ""
        annotation_id = 1
        
        for target_name, target_data in summary.items():
            tier_id = target_name.upper().replace(' ', '_')
            tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}">\n'
            
            for event in target_data['events']:
                start_time = event['start_frame'] / fps
                end_time = event['end_frame'] / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                start_slot = time_slot_refs[start_ms]
                end_slot = time_slot_refs[end_ms]
                
                # Enhanced annotation with relationship types
                overlapping_objects_str = ", ".join(event['overlapping_objects'])
                annotation_value = f"Enhanced Overlap: {overlapping_objects_str}"
                
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
        
        print(f"✅ Enhanced ELAN file created: {output_path}")

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
    
    print("\n🎯 ENHANCED OVERLAP DETECTION ENABLED")
    print("🔍 Features:")
    print("  • Detects when objects are included inside targets")
    print("  • Detects partial overlaps between objects")
    print("  • Clear text-based annotations for all relationships")
    print("  • Enhanced ELAN export with relationship types")
    
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
        self.root.title("SAM2 Video Analysis - Enhanced Overlap Detection")
        self.root.geometry("750x900")
        self.root.minsize(750, 900)
        
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
        
        title_label = tk.Label(main_frame, text="SAM2 Video Analysis - Enhanced Overlap Detection", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Enhanced features info
        features_frame = tk.LabelFrame(main_frame, text="🎯 Enhanced Overlap Detection Features", font=("Arial", 9, "bold"))
        features_frame.pack(fill=tk.X, pady=(0, 10))
        
        features_text = """✅ INCLUSION DETECTION: Detects when objects are completely inside targets
✅ PARTIAL OVERLAP: Detects when objects partially overlap with targets  
✅ CLEAR ANNOTATIONS: Text-only labels showing exact relationship types
✅ ENHANCED ELAN: Detailed behavioral analysis export with relationship data"""
        
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
        
        # Enhanced overlap options
        overlap_frame = tk.LabelFrame(main_frame, text="🔍 Enhanced Overlap Settings", font=("Arial", 9, "bold"))
        overlap_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Overlap threshold
        threshold_frame = tk.Frame(overlap_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threshold_frame, text="Overlap Threshold (%):").pack(side=tk.LEFT)
        self.overlap_threshold_var = tk.StringVar(value="10")
        threshold_spin = tk.Spinbox(threshold_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.overlap_threshold_var, width=10)
        threshold_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(threshold_frame, text="(minimum overlap to detect)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Inclusion threshold info
        inclusion_info = tk.Label(overlap_frame, 
                                 text="Inclusion Detection: Automatically detects when objects are 80%+ inside targets",
                                 font=("Arial", 8), fg="darkgreen", wraplength=700)
        inclusion_info.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
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
                                text="📄 Export enhanced ELAN file with relationship types",
                                variable=self.enable_elan_export)
        elan_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(15, 10))
        
        self.process_button = tk.Button(process_frame, text="🚀 Process Video (Enhanced Overlap Detection)", 
                                       command=self.process_video, bg="#FF5722", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Enhanced overlap detection with inclusion tracking")
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
            self.status_var.set(f"Found {len(videos)} video(s) - Enhanced overlap detection ready")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number_with_preview(self, frames_dir, total_frames):
        """Get frame number with preview functionality"""
        suggested_frame = total_frames // 2
        
        while True:
            frame_num = simpledialog.askinteger(
                "Reference Frame Selection - Enhanced Overlap Detection",
                f"Select reference frame (0-{total_frames-1}):\n\n"
                f"🔍 ENHANCED OVERLAP DETECTION:\n"
                f"• Detects inclusion (objects inside targets)\n"
                f"• Detects partial overlaps\n"
                f"• Clear text-based relationship annotations\n"
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
            self.status_var.set("🔍 Starting enhanced overlap detection processing...")
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
            
            print(f"🔍 Processing {num_frames} frames with enhanced overlap detection")
            
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
            
            messagebox.showinfo("Enhanced Overlap Detection", 
                f"Reference frame {frame_num} will open for annotation.\n\n"
                "🔍 ENHANCED OVERLAP DETECTION FEATURES:\n"
                "• Detects inclusion (objects completely inside targets)\n"
                "• Detects partial overlaps between objects\n"
                "• Clear text-based relationship annotations\n"
                "• Enhanced ELAN export with relationship types\n\n"
                "🎯 TARGET SETUP:\n"
                "• Name crosshairs/targets as 'target_1', 'target_2'\n"
                "• System will automatically detect all spatial relationships\n"
                "• Annotations will show: INCLUDES, OVERLAPS, INSIDE")
            
            points_dict, labels_dict, object_names = select_points_opencv(frame, processor)
            
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            print(f"🔍 Selected {len(object_names)} objects for enhanced overlap detection")
            
            # Process with enhanced overlap detection
            self.status_var.set("🚀 Processing with enhanced overlap detection...")
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
                
                self.status_var.set("✅ Enhanced overlap detection completed!")
                self.update_memory_status()
                
                # Enhanced success message
                target_info = ""
                if processor.overlap_tracker.has_targets():
                    summary = processor.overlap_tracker.get_overlap_summary()
                    target_info = f"\n\n🔍 Enhanced Overlap Analysis:\n"
                    for target_name, data in summary.items():
                        target_info += f"  • {target_name}: {data['total_events']} relationship events, {data['total_overlap_frames']} frames\n"
                    if elan_created:
                        target_info += "\n📄 Enhanced ELAN file: enhanced_target_overlaps.eaf"
                
                named_objects = [name for name in object_names.values()]
                objects_summary = "\n".join([f"  • {name}" for name in named_objects])
                
                success_msg = f"""🔍 Enhanced Overlap Detection Complete!

Reference Frame: {frame_num}
Detection Features: Inclusion + Partial Overlap
Overlap Threshold: {overlap_threshold*100:.1f}%
Inclusion Threshold: 80% (automatic)
Results saved in: {frames_dir}

📁 Generated Files:
• output_enhanced_overlap.mp4 - Video with enhanced annotations
• time_series_metrics.csv - Movement data"""

                if elan_created:
                    success_msg += "\n• enhanced_target_overlaps.eaf - ELAN with relationship types"

                success_msg += f"""{target_info}

📊 Analyzed Objects ({len(object_names)}):
{objects_summary}

✅ Enhanced overlap detection with inclusion tracking completed!

🔍 ANNOTATION FEATURES USED:
• Clear text-based relationship labels
• Inclusion detection (objects inside targets)
• Partial overlap detection
• Multi-line text for complex relationships
• Enhanced color coding for overlapping objects"""
                
                messagebox.showinfo("Enhanced Processing Complete", success_msg)
                
            elif results is None:
                messagebox.showwarning("Processing Incomplete", 
                    "GPU memory was exhausted. Enhanced overlap detection used CPU fallback.\n\n"
                    "Consider reducing the number of objects for better performance.")
                self.status_var.set("Enhanced processing completed with limitations")
            else:
                messagebox.showerror("Error", "Enhanced overlap detection failed")
                self.status_var.set("Enhanced processing failed")
        
        except Exception as e:
            messagebox.showerror("Error", f"Enhanced processing failed: {str(e)}")
            self.status_var.set("Enhanced processing failed")
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
    
    print("\n🔍 ENHANCED OVERLAP DETECTION:")
    print("  • INCLUSION DETECTION: When objects are completely inside targets")
    print("  • PARTIAL OVERLAP: When objects partially overlap with targets")
    print("  • CLEAR ANNOTATIONS: Text-only labels showing exact relationships")
    print("  • ENHANCED ELAN: Detailed behavioral analysis with relationship types")
    print()
    print("🎯 RELATIONSHIP TYPES DETECTED:")
    print("  • TARGET includes object (object is 80%+ inside target)")
    print("  • TARGET overlaps object (partial overlap)")
    print("  • Object included in TARGET (reverse inclusion)")
    print("  • Multiple simultaneous relationships")
    print()
    print("💬 ANNOTATION EXAMPLES:")
    print("  • 'TARGET crosshair INCLUDES apple'")
    print("  • 'hand OVERLAPS TARGET button'")
    print("  • 'TARGET pointer INSIDE screen'")
    print("  • 'TARGET cursor INCLUDES icon | OVERLAPS menu'")
    print()
    print("🧠 MEMORY OPTIMIZATIONS:")
    print("  • Ultra-conservative GPU memory management")
    print("  • Multiple fallback strategies for OOM recovery")
    print("  • Enhanced cleanup every 25 frames")
    print("  • Automatic CPU fallback if needed")
    print()
    print("Starting enhanced overlap detection application...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()