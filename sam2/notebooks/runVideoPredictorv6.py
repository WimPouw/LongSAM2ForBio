#!/usr/bin/env python3
"""
Conservative SAM2 Video Analysis - MINIMAL INTERFERENCE VERSION
FOCUS: Let SAM2 do its job with minimal interruption, robust initial points, proper history tracking
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

# Training-related imports
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
    class Dataset:
        pass

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

class TargetOverlapTracker:
    """Tracks inclusion of objects within ring targets"""
    
    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}
        self.target_objects = {}
        
    def register_target(self, obj_id, obj_name):
        """Register an object as a target if it has 'target' in the name"""
        if 'target' in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"🎯 Registered ring target: {obj_name} (ID: {obj_id})")
            return True
        return False
    
    def calculate_ring_inclusion(self, ring_mask, object_mask):
        """Calculate if ANY part of an object is INSIDE the ring area"""
        if ring_mask.shape != object_mask.shape:
            return 0.0
        
        # Convert masks to proper format
        if len(ring_mask.shape) == 3:
            ring_mask = ring_mask[0]
        if len(object_mask.shape) == 3:
            object_mask = object_mask[0]
            
        ring_mask = ring_mask.astype(bool)
        object_mask = object_mask.astype(bool)
        
        # For ring targets: find the outer boundary and create filled area
        ring_uint8 = ring_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(ring_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest outer contour (should be the ring's outer boundary)
        outer_contour = max(contours, key=cv2.contourArea)
        
        # Create a filled mask representing the entire area enclosed by the ring
        ring_enclosed_area = np.zeros_like(ring_mask, dtype=np.uint8)
        cv2.fillPoly(ring_enclosed_area, [outer_contour], 255)
        ring_enclosed_bool = ring_enclosed_area.astype(bool)
        
        # Calculate how much of the object is within the ring's enclosed area
        object_inside_ring = np.logical_and(ring_enclosed_bool, object_mask)
        inclusion_area = np.sum(object_inside_ring)
        
        # Get total areas for different inclusion strategies
        object_total_area = np.sum(object_mask)
        ring_total_area = np.sum(ring_enclosed_bool)
        
        if object_total_area == 0 or ring_total_area == 0:
            return 0.0
        
        # Use object percentage for inclusion calculation
        percentage_of_object = inclusion_area / object_total_area
        
        # Boost inclusion percentage for meaningful intersections
        min_pixels_for_inclusion = min(50, object_total_area * 0.01)
        if inclusion_area >= min_pixels_for_inclusion:
            percentage_of_object = max(percentage_of_object, 0.1)  # Minimum 10% if meaningful intersection
        
        return min(1.0, percentage_of_object)
    
    def track_frame_overlaps(self, frame_idx, frame_results, object_names):
        """Track ring inclusion for targets"""
        if not self.target_objects:
            return
        
        # Track inclusion for each ring target
        for target_id in self.target_objects:
            target_name = self.target_objects[target_id]
            included_objects = []
            
            # Only check for inclusions if the target exists in this frame
            if target_id in frame_results:
                target_mask = frame_results[target_id]
                
                # Check ALL other objects for inclusion within this ring target
                for obj_id, mask in frame_results.items():
                    if obj_id == target_id:  # Skip the target itself
                        continue
                    
                    # Use corrected ring inclusion detection
                    inclusion_pct = self.calculate_ring_inclusion(target_mask, mask)
                    
                    if inclusion_pct >= self.overlap_threshold:
                        obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                        included_objects.append({
                            'object_id': obj_id,
                            'object_name': obj_name,
                            'overlap_percentage': inclusion_pct
                        })
            
            # Update the event state
            self._update_overlap_event(target_id, frame_idx, included_objects)
    
    def get_frame_overlaps(self, frame_idx, frame_results, object_names):
        """Get current frame inclusions for ring targets"""
        frame_inclusions = {}
        
        if not self.target_objects or not frame_results:
            return frame_inclusions
        
        # Check each ring target for objects inside it
        for target_id in self.target_objects:
            if target_id not in frame_results:
                continue
            
            target_mask = frame_results[target_id]
            included_objects = []
            
            # Check all other objects for inclusion within this ring target
            for obj_id, mask in frame_results.items():
                if obj_id == target_id:  # Skip the target itself
                    continue
                
                # Use corrected ring inclusion detection
                inclusion_pct = self.calculate_ring_inclusion(target_mask, mask)
                
                if inclusion_pct >= self.overlap_threshold:
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    included_objects.append({
                        'object_id': obj_id,
                        'object_name': obj_name,
                        'overlap_percentage': inclusion_pct
                    })
            
            if included_objects:
                frame_inclusions[target_id] = included_objects
        
        return frame_inclusions
    
    def _update_overlap_event(self, target_id, frame_idx, overlapping_objects):
        """Update or create inclusion event"""
        events = self.overlap_events[target_id]
        
        # Get current overlapping object names
        current_overlap_names = set(obj['object_name'] for obj in overlapping_objects) if overlapping_objects else set()
        
        # Check if we have an open event
        if events and not events[-1].get('end_frame'):
            last_event = events[-1]
            last_overlap_names = set(last_event['overlapping_objects'])
            
            if current_overlap_names == last_overlap_names and current_overlap_names:
                # Same objects still overlapping - continue the event
                last_event['end_frame'] = frame_idx
                last_event['duration_frames'] = frame_idx - last_event['start_frame'] + 1
                return
            else:
                # Objects changed or no more overlaps - close the previous event
                last_event['end_frame'] = frame_idx - 1
                last_event['duration_frames'] = last_event['end_frame'] - last_event['start_frame'] + 1
        
        # Start a new event only if we have overlapping objects
        if current_overlap_names:
            new_event = {
                'start_frame': frame_idx,
                'end_frame': None,
                'duration_frames': 1,
                'overlapping_objects': list(current_overlap_names),
                'overlap_details': overlapping_objects
            }
            
            events.append(new_event)
    
    def finalize_tracking(self, last_frame_idx):
        """Finalize any open inclusion events"""
        for target_id, events in self.overlap_events.items():
            if events and not events[-1].get('end_frame'):
                events[-1]['end_frame'] = last_frame_idx
                events[-1]['duration_frames'] = last_frame_idx - events[-1]['start_frame'] + 1
    
    def get_overlap_summary(self):
        """Get summary of all inclusion events"""
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
        """Check if any ring targets are registered"""
        return bool(self.target_objects)

def create_elan_file_with_targets(video_path, overlap_tracker, output_path, fps, object_names=None):
    """Create ELAN file from target inclusion events"""
    if not overlap_tracker.has_targets():
        print("ℹ️  No targets found - skipping ELAN export")
        return
    
    print(f"📄 Creating ELAN file: {output_path}")
    
    summary = overlap_tracker.get_overlap_summary()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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

    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}
    all_time_points = set()
    
    # Collect ALL time points from inclusion events
    for target_name, target_data in summary.items():
        for event in target_data['events']:
            if event.get('start_frame') is not None and event.get('end_frame') is not None:
                start_time = event['start_frame'] / fps
                end_time = event['end_frame'] / fps
                all_time_points.add(start_time)
                all_time_points.add(end_time)
    
    if not all_time_points:
        print("⚠️  No inclusion events found - creating empty ELAN file")
        empty_content = header + '''    </TIME_ORDER>
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
</ANNOTATION_DOCUMENT>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(empty_content)
        print(f"📄 Empty ELAN file created: {output_path}")
        return
    
    # Create time slots for all unique time points
    for time_point in sorted(all_time_points):
        time_ms = int(time_point * 1000)
        time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
        time_slot_refs[time_ms] = f"ts{time_slot_id}"
        time_slot_id += 1

    header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

    tier_content = ""
    annotation_id = 1
    
    # Create tiers for each target with inclusion events
    for target_name, target_data in summary.items():
        if target_data['total_events'] == 0:
            continue
            
        tier_id = target_name.upper().replace(' ', '_').replace('-', '_').replace('.', '_')
        tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}">\n'
        
        # Add annotations for each inclusion event
        for event in target_data['events']:
            if event.get('start_frame') is None or event.get('end_frame') is None:
                continue
                
            start_time = event['start_frame'] / fps
            end_time = event['end_frame'] / fps
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            if start_ms not in time_slot_refs or end_ms not in time_slot_refs:
                continue
                
            start_slot = time_slot_refs[start_ms]
            end_slot = time_slot_refs[end_ms]
            
            # Create detailed annotation value for ring inclusion
            included_objects_str = ", ".join(event['overlapping_objects'])
            duration_frames = event.get('duration_frames', end_time - start_time)
            annotation_value = f"INCLUSION: Objects inside {target_name} → {included_objects_str} (Duration: {duration_frames} frames)"
            
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

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + tier_content + footer)
        
        print(f"✅ ELAN file created: {output_path}")
        
        # Enhanced summary
        print("\n📊 Target Inclusion Summary for ELAN:")
        total_events = 0
        total_frames = 0
        total_duration = 0
        
        for target_name, target_data in summary.items():
            events = target_data['total_events']
            frames = target_data['total_overlap_frames']
            duration_seconds = frames / fps
            total_events += events
            total_frames += frames
            total_duration += duration_seconds
            
            if events > 0:
                avg_event_duration = frames / events
                print(f"  {target_name}:")
                print(f"    • {events} inclusion events")
                print(f"    • {frames} total frames ({duration_seconds:.2f}s)")
                print(f"    • Average event: {avg_event_duration:.1f} frames ({avg_event_duration/fps:.2f}s)")
        
        print(f"  📊 TOTAL SUMMARY:")
        print(f"    • {total_events} inclusion events across all targets")
        print(f"    • {total_frames} total inclusion frames ({total_duration:.2f}s)")
        if total_events > 0:
            print(f"    • Average event duration: {total_frames/total_events:.1f} frames ({total_duration/total_events:.2f}s)")
        
    except Exception as e:
        print(f"❌ Error creating ELAN file: {e}")
        import traceback
        traceback.print_exc()

class VideoChunkProcessor:
    def __init__(self, predictor, video_dir, chunk_size=500, overlap_frames=20, 
             interactive_correction=True, seed_frame_idx=0, overlap_threshold=0.1):
        self.predictor = predictor
        self.video_dir = video_dir
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.interactive_correction = interactive_correction
        self.seed_frame_idx = seed_frame_idx
        
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
        
        if self.seed_frame_idx >= len(self.frame_names):
            self.seed_frame_idx = len(self.frame_names) // 2
            print(f"⚠️ Seed frame index too high, using middle frame: {self.seed_frame_idx}")
        
        # CONSERVATIVE: Proper mask history tracking
        self.mask_quality_history = {}  # obj_id -> [(frame_idx, mask, area, quality)]
        self.global_results_history = {}  # frame_idx -> {obj_id: mask}
        
        print(f"Created CONSERVATIVE processor with chunk size {chunk_size}, seed frame at {self.seed_frame_idx}")
        print(f"🚀 FOCUS: Let SAM2 work naturally with minimal interference")

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
    
    def _check_mask_quality(self, mask, min_area=50):
        """LENIENT mask quality check"""
        if mask is None:
            return False, "MISSING", 0

        if len(mask.shape) > 2:
            mask = mask[0]

        area = np.sum(mask.astype(bool))

        if area == 0:
            return False, "EMPTY", 0
        elif area < min_area:
            return False, "SMALL", area
        else:
            return True, "OK", area
    
    def _update_mask_history(self, frame_idx, frame_masks):
        """FIXED: Properly update mask history for all objects"""
        # Store global results
        self.global_results_history[frame_idx] = {}
        
        for obj_id, mask in frame_masks.items():
            quality_ok, issue, area = self._check_mask_quality(mask)
            
            # Store in global history
            self.global_results_history[frame_idx][obj_id] = mask.copy()
            
            # Update object-specific history
            if obj_id not in self.mask_quality_history:
                self.mask_quality_history[obj_id] = []
            
            self.mask_quality_history[obj_id].append({
                'frame_idx': frame_idx,
                'mask': mask.copy(),
                'area': area,
                'quality_ok': quality_ok,
                'issue': issue
            })
            
            # Keep history manageable (last 50 frames per object)
            if len(self.mask_quality_history[obj_id]) > 50:
                self.mask_quality_history[obj_id] = self.mask_quality_history[obj_id][-50:]
        
        # Also add entries for missing objects (helps with detection)
        for obj_id in self.object_names:
            if obj_id not in frame_masks:
                if obj_id not in self.mask_quality_history:
                    self.mask_quality_history[obj_id] = []
                
                self.mask_quality_history[obj_id].append({
                    'frame_idx': frame_idx,
                    'mask': None,
                    'area': 0,
                    'quality_ok': False,
                    'issue': 'MISSING'
                })
                
                if len(self.mask_quality_history[obj_id]) > 50:
                    self.mask_quality_history[obj_id] = self.mask_quality_history[obj_id][-50:]
        
    def _find_previous_good_mask(self, obj_id):
        """FIXED: Find the most recent good mask for an object"""
        if obj_id not in self.mask_quality_history:
            print(f"  🔍 No history for object {obj_id}")
            return None, None
        
        history = self.mask_quality_history[obj_id]
        print(f"  🔍 Searching {len(history)} history entries for object {obj_id}")
        
        # Look through history from most recent backwards
        for entry in reversed(history):
            if entry['quality_ok'] and entry['mask'] is not None and entry['area'] > 100:
                print(f"  ✅ Found good mask for object {obj_id} from frame {entry['frame_idx']} (area: {entry['area']})")
                return entry['mask'].copy(), entry['frame_idx']
        
        print(f"  ❌ No good previous mask found for object {obj_id}")
        return None, None
        
    def _generate_robust_points_from_mask(self, mask, num_positive=12, num_negative=8):
        """Generate MORE robust points from mask for better propagation"""
        if not mask.any():
            return None, None
            
        points = []
        labels = []
        
        if len(mask.shape) == 3:
            mask = mask[0]
        mask = mask.astype(bool)
        
        # Get multiple points from contours - MORE POINTS for robustness
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get points from the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_length = cv2.arcLength(largest_contour, True)
            if contour_length > 0:
                # More evenly distributed points
                spacing = max(1, int(contour_length / (num_positive * 1.5)))
                for i in range(0, len(largest_contour), spacing):
                    if len([p for p, l in zip(points, labels) if l == 1]) >= num_positive:
                        break
                    point = largest_contour[i][0]
                    points.append([point[0], point[1]])
                    labels.append(1)
        
        # Add center points - MULTIPLE center points for robustness
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            if mask[cy, cx]:
                points.append([cx, cy])
                labels.append(1)
        
        # Add points from eroded mask (more interior points)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        if np.sum(eroded) > 0:
            interior_points = np.where(eroded)
            if len(interior_points[0]) > 0:
                # Add several interior points
                for _ in range(min(3, len(interior_points[0]))):
                    idx = np.random.choice(len(interior_points[0]))
                    y, x = interior_points[0][idx], interior_points[1][idx]
                    points.append([x, y])
                    labels.append(1)
        
        # Add negative points - FEWER negative points to avoid confusion
        kernel_size = max(15, int(np.sqrt(np.sum(mask)) * 0.15))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        negative_region = expanded & (~mask)
        
        neg_y, neg_x = np.where(negative_region)
        if len(neg_x) > 0:
            neg_indices = np.random.choice(len(neg_x), min(num_negative, len(neg_x)), replace=False)
            for idx in neg_indices:
                points.append([neg_x[idx], neg_y[idx]])
                labels.append(0)
        
        if not points:
            return None, None
            
        print(f"  🎯 Generated {sum(labels)} positive + {len(labels) - sum(labels)} negative = {len(points)} total points")
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _minimal_quality_check(self, current_masks, frame_idx):
        """EXTREMELY CONSERVATIVE quality check - almost never triggers"""
        expected_objects = len(self.object_names)
        if expected_objects == 0:
            return False
        
        # Only trigger if 100% of objects are completely missing for 3+ consecutive frames
        completely_missing = sum(1 for obj_id in self.object_names if obj_id not in current_masks)
        
        if completely_missing == expected_objects:  # ALL objects missing
            # Check if this is consecutive
            consecutive_missing = 1
            for check_frame in range(frame_idx - 1, max(frame_idx - 5, 0), -1):
                if check_frame in self.global_results_history:
                    frame_missing = sum(1 for obj_id in self.object_names 
                                      if obj_id not in self.global_results_history[check_frame])
                    if frame_missing == expected_objects:
                        consecutive_missing += 1
                    else:
                        break
                else:
                    break
            
            if consecutive_missing >= 3:
                print(f"⚠️  EMERGENCY: ALL {expected_objects} objects missing for {consecutive_missing} consecutive frames!")
                return True
        
        return False

    def _emergency_correction_interface(self, current_frame_idx, frame_path, current_masks):
        """MINIMAL emergency correction - only when ALL objects lost"""
        
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        
        # Simple choice: restore all or manual points
        choice = messagebox.askyesnocancel(
            "EMERGENCY: All Objects Lost", 
            f"Frame {current_frame_idx}: ALL objects have been lost for multiple frames.\n\n"
            f"Options:\n"
            f"YES = Try to restore all objects from previous good masks\n"
            f"NO = Add manual correction points\n"
            f"CANCEL = Continue without correction (may lose tracking)\n\n"
            f"Recommendation: Try RESTORE first"
        )
        
        root.destroy()
        
        if choice is None:  # Cancel
            return current_masks
        
        elif choice is True:  # Restore all
            print("🔧 Attempting to restore all objects from previous masks...")
            restored_masks = current_masks.copy()
            restore_count = 0
            
            for obj_id in self.object_names:
                previous_mask, previous_frame = self._find_previous_good_mask(obj_id)
                if previous_mask is not None:
                    restored_masks[obj_id] = previous_mask.copy()
                    restore_count += 1
                    obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                    print(f"  ✅ Restored {obj_name} from frame {previous_frame}")
            
            print(f"📊 Restored {restore_count}/{len(self.object_names)} objects")
            return restored_masks
        
        else:  # Manual points
            frame = cv2.imread(frame_path)
            print("🔧 Opening manual point correction...")
            
            points_dict, labels_dict, updated_names = select_points_opencv(
                frame, self,
                current_masks=current_masks,
                object_names=self.object_names,
                current_frame_idx=current_frame_idx,
                correction_mode=True
            )
            
            if points_dict is not None:
                corrected_masks = self._apply_point_corrections(frame, points_dict, labels_dict, current_frame_idx)
                return corrected_masks
            
            return current_masks

    def _apply_point_corrections(self, frame, points_dict, labels_dict, frame_idx):
        """Apply point-based corrections using SAM2"""
        corrected_masks = {}
        temp_dir = f"temp_correction_{frame_idx}"
        
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            correction_frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(correction_frame_path, frame)
            
            correction_state = self.predictor.init_state(video_path=temp_dir)
            
            for obj_id in points_dict:
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
                    
                    for i, corrected_obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        
                        corrected_masks[corrected_obj_id] = mask.copy()
                        new_area = np.sum(mask.astype(bool))
                        
                        print(f"✅ Point-corrected {self.object_names.get(corrected_obj_id, f'Object_{corrected_obj_id}')} - new area: {new_area} pixels")
                    
                    del mask_logits
                    cleanup_memory()
                
                except Exception as e:
                    print(f"Error correcting object {obj_id}: {e}")
                    continue
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return corrected_masks

    def process_video(self, points_dict, labels_dict, debug=True):
        """CONSERVATIVE processing - let SAM2 work with minimal interference"""
        results = {}
        
        try:
            cleanup_memory()
            
            self.object_names = getattr(self, 'object_names', {})
            
            print(f"\n🚀 CONSERVATIVE Processing - Minimal Interference Mode")
            print(f"Step 1: Processing seed frame {self.seed_frame_idx}")
            seed_results = self._process_seed_frame(points_dict, labels_dict, debug)
            
            if not seed_results:
                print("❌ Failed to process seed frame")
                return None
            
            results.update(seed_results)
            # Update history with seed results
            self._update_mask_history(self.seed_frame_idx, seed_results[self.seed_frame_idx])
            print(f"✅ Seed processing complete")
            
            print(f"\n➡️ Step 2: Forward propagation (minimal interference)")
            forward_results = self._process_forward_propagation(seed_results, debug)
            results.update(forward_results)
            print(f"✅ Forward propagation complete: {len(forward_results)} frames")
            
            print(f"\n⬅️ Step 3: Backward propagation (minimal interference)")
            backward_results = self._process_backward_propagation(seed_results, debug)
            results.update(backward_results)
            print(f"✅ Backward propagation complete: {len(backward_results)} frames")
            
            self._fill_result_gaps(results, debug)
            print(f"\n🎉 CONSERVATIVE processing complete! Total frames: {len(results)}/{len(self.frame_names)}")
            
            if hasattr(self, 'overlap_tracker') and self.object_names:
                targets_found = False
                for obj_id, obj_name in self.object_names.items():
                    if self.overlap_tracker.register_target(obj_id, obj_name):
                        targets_found = True
                
                if targets_found:
                    print(f"\nTracking target inclusions (threshold: {self.overlap_threshold*100:.1f}%)...")
                    
                    # Track ALL frames
                    for frame_idx in tqdm(range(len(self.frame_names)), desc="Tracking inclusions"):
                        # Get results for this frame (empty dict if frame not processed)
                        frame_results = results.get(frame_idx, {})
                        
                        # Always call tracking
                        self.overlap_tracker.track_frame_overlaps(frame_idx, frame_results, self.object_names)
                    
                    print(f"✅ Target inclusion tracking completed")
                    
                    summary = self.overlap_tracker.get_overlap_summary()
                    print("\n📊 Inclusion Summary:")
                    for target_name, data in summary.items():
                        print(f" {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames")
                else:
                    print("ℹ️  No target objects found (use 'target_1', 'target_2', etc. for inclusion tracking)")
            
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
            seed_frame_name = self.frame_names[self.seed_frame_idx]
            temp_dir_path = self._create_temp_video_dir([seed_frame_name], temp_dir)
            
            chunk_state = self.predictor.init_state(video_path=temp_dir_path)
            
            for obj_id in points_dict:
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    points = np.array(points_dict[obj_id], dtype=np.float32)
                    labels = np.array(labels_dict[obj_id], dtype=np.int32)
                    
                    if debug:
                        print(f"  Object {obj_id}: +{sum(labels == 1)} -{sum(labels == 0)} points")
                    
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
        
        forward_frames = self.frame_names[self.seed_frame_idx + 1:]
        if not forward_frames:
            return forward_results
        
        for chunk_start in range(0, len(forward_frames), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(forward_frames))
            chunk_frames = forward_frames[chunk_start:chunk_end]
            
            chunk_with_seed = [self.frame_names[self.seed_frame_idx]] + chunk_frames
            
            if debug:
                print(f"  Forward chunk: frames {self.seed_frame_idx + 1 + chunk_start} to {self.seed_frame_idx + chunk_end}")
            
            chunk_results = self._process_chunk_conservative(chunk_with_seed, seed_results, is_forward=True, debug=debug)
            
            chunk_results.pop(self.seed_frame_idx, None)
            forward_results.update(chunk_results)
        
        return forward_results

    def _process_backward_propagation(self, seed_results, debug=True):
        """Process frames backward from seed to start"""
        backward_results = {}
        
        backward_frames = self.frame_names[:self.seed_frame_idx]
        if not backward_frames:
            return backward_results
        
        backward_frames_reversed = backward_frames[::-1]
        
        for chunk_start in range(0, len(backward_frames_reversed), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(backward_frames_reversed))
            chunk_frames = backward_frames_reversed[chunk_start:chunk_end]
            
            chunk_with_seed = [self.frame_names[self.seed_frame_idx]] + chunk_frames
            
            if debug:
                original_indices = [self.seed_frame_idx - 1 - chunk_start - i for i in range(len(chunk_frames))]
                print(f"  Backward chunk: frames {min(original_indices)} to {max(original_indices)}")
            
            chunk_results = self._process_chunk_conservative(chunk_with_seed, seed_results, is_forward=False, debug=debug)
            
            chunk_results.pop(self.seed_frame_idx, None)
            backward_results.update(chunk_results)
        
        return backward_results
    
    def _process_chunk_conservative(self, chunk_frames, reference_results, is_forward=True, debug=True):
        """CONSERVATIVE chunk processing - minimal interference with SAM2"""
        chunk_results = {}
        temp_dir = f"temp_{'forward' if is_forward else 'backward'}"
        
        try:
            temp_dir_path = self._create_temp_video_dir(chunk_frames, temp_dir)
            chunk_state = self.predictor.init_state(video_path=temp_dir_path)
            
            seed_masks = reference_results.get(self.seed_frame_idx, {})
            
            for obj_id, reference_mask in seed_masks.items():
                try:
                    self.predictor.reset_state(chunk_state)
                    
                    # Generate ROBUST points for better propagation
                    points, labels = self._generate_robust_points_from_mask(reference_mask)
                    if points is None:
                        continue
                    
                    _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=chunk_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
                    
                    del mask_logits
                    cleanup_memory()
                    
                    frame_count = 0
                    
                    # LET SAM2 DO ITS JOB - minimal interference
                    for local_frame_idx, prop_obj_ids, prop_mask_logits in self.predictor.propagate_in_video(chunk_state):
                        if local_frame_idx == 0:
                            continue
                        
                        global_frame_idx = self._map_local_to_global_index(local_frame_idx, chunk_frames, is_forward)
                        
                        if global_frame_idx is None:
                            continue
                        
                        frame_masks = {}
                        for i, prop_obj_id in enumerate(prop_obj_ids):
                            mask = (prop_mask_logits[i] > 0.0).cpu().numpy()
                            if len(mask.shape) == 3:
                                mask = mask[0]
                            frame_masks[prop_obj_id] = mask.copy()
                        
                        # Update mask history EVERY frame
                        self._update_mask_history(global_frame_idx, frame_masks)
                        
                        # EXTREMELY CONSERVATIVE quality check - almost never triggers
                        if (self.interactive_correction and frame_count > 0 and 
                            frame_count % 1000 == 0):  # Only check every 1000 frames
                            
                            emergency_needed = self._minimal_quality_check(frame_masks, global_frame_idx)
                            
                            if emergency_needed:
                                frame_path = os.path.join(self.video_dir, self.frame_names[global_frame_idx])
                                corrected_masks = self._emergency_correction_interface(
                                    global_frame_idx, frame_path, frame_masks
                                )
                                
                                if corrected_masks is not frame_masks:
                                    print(f"🔧 Emergency correction applied at frame {global_frame_idx}")
                                    # Update history with corrected masks
                                    self._update_mask_history(global_frame_idx, corrected_masks)
                                    frame_masks = corrected_masks
                        
                        # Store results
                        if global_frame_idx not in chunk_results:
                            chunk_results[global_frame_idx] = {}
                        
                        for prop_obj_id, mask in frame_masks.items():
                            chunk_results[global_frame_idx][prop_obj_id] = mask
                        
                        del prop_mask_logits
                        cleanup_memory()
                        frame_count += 1
                        
                        # Progress feedback
                        if frame_count % 500 == 0:
                            print(f"  📍 {['Backward', 'Forward'][is_forward]} progress: frame {global_frame_idx} (+{frame_count})")
                
                except Exception as e:
                    if debug:
                        print(f"    Error processing object {obj_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            return chunk_results
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
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
        """Create enhanced analysis video with ring inclusion indicators"""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        if not results:
            print("No results to analyze!")
            return
        
        cmap = plt.get_cmap("tab10")
        object_names = getattr(self, 'object_names', {})
        overlap_tracker = getattr(self, 'overlap_tracker', None)
        
        target_objects = {}
        if overlap_tracker and overlap_tracker.has_targets():
            target_objects = overlap_tracker.target_objects
            print(f"Creating analysis video with {len(target_objects)} ring targets: {list(target_objects.values())}")
        
        print("Collecting time series data...")
        time_series_data = {}
        max_frame_idx = max(results.keys())
        
        for obj_id in set(obj_id for frame in results.values() for obj_id in frame.keys()):
            time_series_data[obj_id] = {
                'frames': [],
                'centroids': [],
                'areas': [],
                'plot_color': cmap(obj_id % 10)[:3],
                'is_target': obj_id in target_objects
            }
        
        print("Calculating metrics...")
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
        
        print("Calculating derived metrics...")
        window_size = 10
        for obj_id in time_series_data:
            data = time_series_data[obj_id]
            
            if not data['frames']:
                continue
            
            centroids = np.array(data['centroids'])
            if len(centroids) > 1:
                data['movement'] = np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))
                data['movement'] = np.insert(data['movement'], 0, 0)
            else:
                data['movement'] = np.array([0])
            
            data['area_ma'] = np.convolve(data['areas'], 
                                        np.ones(window_size)/window_size, 
                                        mode='same')
            data['movement_ma'] = np.convolve(data['movement'],
                                            np.ones(window_size)/window_size,
                                            mode='same')
        
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]
        
        n_objects = len([obj_id for obj_id in time_series_data if time_series_data[obj_id]['frames']])
        if n_objects == 0:
            print("No valid objects for analysis video")
            return
        
        side_plot_height = height // max(n_objects, 1)
        side_plot_width = width // 3
        
        out_width = width + (2 * side_plot_width)
        out_height = height
        
        video_x = side_plot_width
        video_y = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        print("\nCreating enhanced analysis video with ring inclusion detection...")
        inclusion_frame_count = 0
        
        for frame_idx in tqdm(range(len(self.frame_names)), desc="Creating analysis video"):
            output_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            overlay = frame.copy()
            
            centroids = {}
            
            frame_inclusions = {}
            has_inclusions = False
            
            if overlap_tracker and overlap_tracker.has_targets() and frame_idx in results:
                frame_inclusions = overlap_tracker.get_frame_overlaps(frame_idx, results[frame_idx], object_names)
                has_inclusions = bool(frame_inclusions)
                if has_inclusions:
                    inclusion_frame_count += 1
            
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
                    
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        centroids[obj_id] = (cx + video_x, cy + video_y)
                    
                    is_inside_ring = False
                    containing_rings = []
                    
                    # Check if this object is included in any ring target
                    for target_id, inclusion_list in frame_inclusions.items():
                        for inclusion_info in inclusion_list:
                            if inclusion_info['object_id'] == obj_id:
                                is_inside_ring = True
                                ring_name = target_objects.get(target_id, f"Ring_{target_id}")
                                if ring_name not in containing_rings:
                                    containing_rings.append(ring_name)
                    
                    base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                    if is_inside_ring:
                        color = np.minimum(base_color + [0, 80, 0], 255)
                        border_color = (0, 255, 0)
                    elif obj_id in target_objects:
                        color = np.minimum(base_color + [50, 50, 0], 255)
                        border_color = (0, 255, 255)
                    else:
                        color = base_color
                        border_color = None
                    
                    color_mask = np.zeros_like(overlay)
                    for c in range(3):
                        color_mask[:, :, c][mask] = color[c]
                    
                    blend_mask = np.zeros_like(overlay)
                    cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0, blend_mask)
                    overlay[mask] = blend_mask[mask]
                    
                    if border_color:
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, border_color, 3)
                    
                    if centroids.get(obj_id):
                        cx, cy = centroids[obj_id]
                        obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                        
                        if is_inside_ring:
                            if obj_id in target_objects:
                                label = f"🎯{obj_name} [RING]"
                            else:
                                ring_names = ", ".join(containing_rings)
                                label = f"{obj_name} 🟢 INSIDE {ring_names}"
                        else:
                            if obj_id in target_objects:
                                label = f"🎯{obj_name} [RING]"
                            else:
                                label = obj_name
                        
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        label_x = cx - video_x - text_size[0]//2
                        label_y = cy - video_y
                        
                        if is_inside_ring and obj_id not in target_objects:
                            bg_color = (0, 150, 0)
                            text_color = (255, 255, 255)
                        elif obj_id in target_objects:
                            bg_color = (0, 150, 150)
                            text_color = (0, 0, 0)
                        else:
                            bg_color = (0, 0, 0)
                            text_color = (255, 255, 255)
                        
                        cv2.rectangle(overlay, (label_x - 5, label_y - 20), 
                                    (label_x + text_size[0] + 5, label_y + 5), 
                                    bg_color, -1)
                        
                        cv2.putText(overlay, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            output_frame[video_y:video_y+height, video_x:video_x+width] = overlay
            
            # Add plotting functionality (simplified for performance)
            plot_idx = 0
            for obj_id, data in time_series_data.items():
                if not data['frames'] or plot_idx >= n_objects:
                    continue
                
                plot_color = data['plot_color']
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                is_target = data['is_target']
                
                y_offset = plot_idx * side_plot_height
                
                try:
                    fig_left = Figure(figsize=(side_plot_width/100, side_plot_height/100), dpi=100)
                    ax_left = fig_left.add_subplot(111)
                    
                    if len(data['movement']) > 0 and len(data['frames']) > 0:
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
                    
                    plt.close(fig_left)
                    
                except Exception as e:
                    pass  # Skip plotting errors to maintain performance
                
                plot_idx += 1
            
            info_text = f"Frame {frame_idx}/{len(self.frame_names)-1}"
            if has_inclusions:
                info_text += " - RING INCLUSION!"
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.rectangle(output_frame, (5, 5), (len(info_text) * 15, 40), (0, 100, 0), -1)
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(output_frame)
        
        out.release()
        print(f"✅ CONSERVATIVE analysis video with ring inclusion detection saved to: {output_path}")
        print(f"📊 Found ring inclusions in {inclusion_frame_count} frames out of {len(self.frame_names)} total frames")

    def create_simple_analysis_video(self, results, output_path, fps=30, alpha=0.5):
        """Create a simple analysis video without complex plots as fallback"""
        if not results:
            print("No results to analyze!")
            return
            
        print("Creating simple analysis video...")
        self.save_results_video(results, output_path, fps, show_original=True, alpha=alpha)
        print(f"Simple analysis video saved to: {output_path}")

    def save_results_with_elan(self, results, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results including ELAN file if targets are present"""
        self.save_results_video(results, output_path, fps, show_original, alpha)
        self._save_coco_annotations(os.path.join(os.path.dirname(output_path), "segmentation_coco.json"))
        self._save_time_series(os.path.join(os.path.dirname(output_path), "time_series_metrics.csv"))
        
        if hasattr(self, 'overlap_tracker') and self.overlap_tracker and self.overlap_tracker.has_targets():
            elan_path = os.path.join(os.path.dirname(output_path), "target_overlaps.eaf")
            object_names = getattr(self, 'object_names', {})
            
            try:
                create_elan_file_with_targets(
                    video_path=output_path,
                    overlap_tracker=self.overlap_tracker,
                    output_path=elan_path,
                    fps=fps,
                    object_names=object_names
                )
                
                print(f"📄 ELAN file saved: {os.path.basename(elan_path)}")
                
            except Exception as e:
                print(f"❌ Error creating ELAN file: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️  No targets detected - ELAN export skipped")

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
        """Save results as video with ring inclusion indicators"""
        if not results:
            print("No results to save!")
            return

        fps = float(fps)
        alpha = max(0.0, min(1.0, alpha))
        
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]

        cmap = plt.get_cmap("tab10")
        object_names = getattr(self, 'object_names', {})
        overlap_tracker = getattr(self, 'overlap_tracker', None)
        target_objects = {}
        
        if overlap_tracker and overlap_tracker.has_targets():
            target_objects = overlap_tracker.target_objects
        
        def get_object_name(obj_id):
            return object_names.get(obj_id, f"Object_{obj_id}")
        
        if show_original:
            out_width = width * 2
        else:
            out_width = width
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))

        print("Saving video with ring inclusion indicators...")
        for frame_idx in tqdm(range(len(self.frame_names))):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            if frame is None:
                continue
                
            overlay = frame.copy()
            
            # Get ring inclusions for this frame
            frame_inclusions = {}
            if overlap_tracker and overlap_tracker.has_targets() and frame_idx in results:
                frame_inclusions = overlap_tracker.get_frame_overlaps(frame_idx, results[frame_idx], object_names)
            
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
                        # Check if this object is inside any ring target
                        is_inside_ring = False
                        containing_rings = []
                        
                        # Check if this object is included in any ring target
                        for target_id, inclusion_list in frame_inclusions.items():
                            for inclusion_info in inclusion_list:
                                if inclusion_info['object_id'] == obj_id:
                                    is_inside_ring = True
                                    ring_name = target_objects.get(target_id, f"Ring_{target_id}")
                                    if ring_name not in containing_rings:
                                        containing_rings.append(ring_name)
                        
                        # Choose colors based on inclusion status
                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        if is_inside_ring:
                            # Bright green tint for objects inside ring targets
                            color = np.minimum(base_color + [0, 80, 0], 255)
                            border_color = (0, 255, 0)  # Bright green border
                            border_thickness = 4
                        elif obj_id in target_objects:
                            # Special color for ring targets themselves
                            color = np.minimum(base_color + [50, 50, 0], 255)
                            border_color = (0, 255, 255)  # Yellow border for rings
                            border_thickness = 3
                        else:
                            color = base_color
                            border_color = None
                            border_thickness = 1
                        
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
                        
                        # Add border for special cases
                        if border_color:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, border_color, border_thickness)
                        
                        # Add labels
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            name = get_object_name(obj_id)
                            
                            # Enhanced labels for ring inclusion
                            if is_inside_ring:
                                if obj_id in target_objects:
                                    # This is a ring target
                                    display_name = f"🎯{name}"
                                else:
                                    # This object is inside a ring target
                                    display_name = f"{name} 🟢 INSIDE"
                            else:
                                if obj_id in target_objects:
                                    display_name = f"🎯{name}"
                                else:
                                    display_name = name
                            
                            # Add background for better text visibility
                            text_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            if is_inside_ring and obj_id not in target_objects:
                                bg_color = (0, 100, 0)  # Green background for included objects
                                text_color = (255, 255, 255)
                            elif obj_id in target_objects:
                                bg_color = (0, 100, 100)  # Teal background for ring targets
                                text_color = (255, 255, 255)
                            else:
                                bg_color = (0, 0, 0)
                                text_color = (255, 255, 255)
                            
                            cv2.rectangle(overlay, (cx - text_size[0]//2 - 5, cy + 5), 
                                        (cx + text_size[0]//2 + 5, cy + 25), bg_color, -1)
                            
                            cv2.putText(overlay, display_name, (cx - text_size[0]//2, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay
                
            out.write(output_frame)

        out.release()
        print(f"CONSERVATIVE video with ring inclusion indicators saved to: {output_path}")

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
                "description": "SAM2 segmentation results - CONSERVATIVE processing",
                "date_created": current_time
            },
            "images": [],
            "annotations": [],
            "licenses": [{"id": 0, "name": "Unknown License", "url": ""}],
            "categories": []
        }
        
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

# CONSERVATIVE: Enhanced point selection for robust initial tracking
def select_points_opencv(frame, processor=None, current_masks=None, object_names=None, current_frame_idx=None, correction_mode=False):
    """CONSERVATIVE: Enhanced point selection for robust initial tracking"""
    points_dict = {}
    labels_dict = {}
    if object_names is None:
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
    
    def create_display_frame():
        """Create display frame showing current masks + user annotations"""
        display = frame.copy()
        
        # Show CURRENT MASKS if in correction mode
        if current_masks is not None and correction_mode:
            cmap = plt.get_cmap("tab10")
            
            for obj_id, mask in current_masks.items():
                if np.sum(mask.astype(bool)) > 0:
                    # Fixed color handling
                    color = np.array(cmap(obj_id % 10)[:3]) * 255
                    color = color.astype(np.uint8)
                    
                    # Apply very light overlay for existing masks
                    color_mask = np.zeros_like(display)
                    mask_bool = mask.astype(bool)
                    for c in range(3):
                        color_mask[:, :, c][mask_bool] = color[c]
                    display = cv2.addWeighted(display, 0.9, color_mask, 0.1, 0)
                    
                    # Add thin border for existing masks
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    border_color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.drawContours(display, contours, -1, border_color, 1)
                    
                    # Add small label for existing objects
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        existing_name = object_names.get(obj_id, f"Object_{obj_id}")
                        label = f"EXISTS: {existing_name}"
                        
                        cv2.putText(display, label, (cx - 50, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Show USER ANNOTATIONS (new points being added) in bright colors
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        
        return display
    
    def draw_point(img, point, obj_id, label):
        """Draw user annotation points in bright colors"""
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 6, color, -1)  # Slightly larger for visibility
        
        display_name = get_object_name(obj_id)
        cv2.putText(img, display_name, 
                (int(point[0] + 8), int(point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def redraw_all_points():
        """Redraw display with proper mode labeling and enhanced instructions"""
        display = create_display_frame()
        height, width = display.shape[:2]
        
        # Add enhanced instructions overlay
        overlay = display.copy()
        instructions_height = 320
        cv2.rectangle(overlay, (10, height - instructions_height - 10), 
                    (width - 10, height - 10), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Enhanced mode labeling
        if correction_mode:
            mode_title = "🔧 EMERGENCY CORRECTION - Only when ALL objects lost"
            mode_color = (0, 255, 255)  # Cyan for emergency
            mode_tip = "Focus on restoring missing objects"
        else:
            mode_title = "🚀 CONSERVATIVE ANNOTATION - Robust initial tracking"
            mode_color = (0, 255, 0)  # Green for initial
            mode_tip = "Add MORE points for robust SAM2 propagation"
        
        instructions = [
            mode_title,
            "",
            "CONSERVATIVE APPROACH: More points = better tracking",
            "YOUR POINTS: Bright green (+) and red (-) for robust annotation",
            "",
            "KEYBOARD SHORTCUTS:",
            "Left Click: Add positive point (+) - ADD MANY!",
            "Right Click: Add negative point (-) - Use sparingly",
            "R: Reset current object",
            "N: Next object  P: Previous object", 
            "C: Name current object",
            "T: Test/preview mask (RECOMMENDED before finishing)",
            "Enter: Finish  Q: Quit",
            "",
            mode_tip,
            "🎯 For robust tracking: 8-12 positive points per object",
            "💡 Target objects: Use 'target_1', 'target_2' for ring detection"
        ]
        
        if current_frame_idx is not None:
            instructions[0] = f"{mode_title} - Frame {current_frame_idx}"
        
        y_start = height - instructions_height + 10
        for i, instruction in enumerate(instructions):
            if instruction == "":
                continue
                
            if instruction.startswith(("🔧 EMERGENCY", "🚀 CONSERVATIVE")):
                color = mode_color
                font_scale = 0.7
                thickness = 2
            elif instruction.startswith("CONSERVATIVE APPROACH") or instruction.startswith("YOUR POINTS"):
                color = (0, 255, 0)
                font_scale = 0.5
                thickness = 1
            elif instruction.startswith("KEYBOARD"):
                color = (255, 255, 0)
                font_scale = 0.6
                thickness = 2
            elif instruction.startswith(("Focus on", "Add MORE", "🎯 For", "💡 Target")):
                color = (255, 100, 255)
                font_scale = 0.5
                thickness = 1
            else:
                color = (255, 255, 255)
                font_scale = 0.45
                thickness = 1
            
            cv2.putText(display, instruction, (20, y_start + (i * 18)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Show current object info with point count guidance
        current_obj_name = get_object_name(current_obj_id)
        obj_info = f"Current Object: {current_obj_name}"
        cv2.putText(display, obj_info, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if current_obj_id in points_dict:
            pos_count = sum(1 for l in labels_dict[current_obj_id] if l == 1)
            neg_count = sum(1 for l in labels_dict[current_obj_id] if l == 0)
            
            # Color-code based on point count
            if pos_count >= 8:
                count_color = (0, 255, 0)  # Green - good
                status = "EXCELLENT"
            elif pos_count >= 5:
                count_color = (0, 255, 255)  # Yellow - ok
                status = "GOOD"
            else:
                count_color = (0, 0, 255)  # Red - needs more
                status = "NEEDS MORE"
            
            count_info = f"Points: +{pos_count} -{neg_count} ({status})"
            cv2.putText(display, count_info, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2)
        
        # Show summary for correction mode
        if current_masks is not None and correction_mode:
            detected_count = len([obj_id for obj_id, mask in current_masks.items() if np.sum(mask.astype(bool)) > 0])
            total_expected = len(object_names) if object_names else 0
            summary = f"Emergency: {detected_count}/{total_expected} objects detected"
            cv2.putText(display, summary, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
        
        return display
    
    def name_current_object():
        import tkinter as tk
        from tkinter import simpledialog
        
        root = tk.Tk()
        root.withdraw()
        
        current_name = object_names.get(current_obj_id, f"Object_{current_obj_id}")
        
        if correction_mode:
            prompt_text = f"Enter name for object {current_obj_id} (EMERGENCY CORRECTION):"
        else:
            prompt_text = f"Enter name for object {current_obj_id} (CONSERVATIVE MODE):\n\n💡 For robust tracking, use descriptive names\n🎯 Use 'target_1', 'target_2' for ring/crosshair objects"
        
        name = simpledialog.askstring("Object Name", prompt_text, initialvalue=current_name)
        root.destroy()
        
        if name and name.strip():
            object_names[current_obj_id] = name.strip()
            print(f"Object {current_obj_id} named: {object_names[current_obj_id]}")
            
            if 'target' in name.lower():
                print(f"🎯 Detected target object: {name} - will enable ring inclusion tracking")
            
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
            
            pos_count = sum(labels == 1)
            neg_count = sum(labels == 0)
            
            print(f"Testing mask with {pos_count} positive and {neg_count} negative points...")
            
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
                
                # Create preview showing current masks + new preview
                preview = create_display_frame()
                
                # Add the new mask preview in bright yellow
                color_overlay = np.zeros_like(preview)
                color_overlay[:, :, 1][mask] = 255  # Yellow channel
                color_overlay[:, :, 2][mask] = 255  # Yellow = Green + Red
                
                preview = cv2.addWeighted(preview, 0.7, color_overlay, 0.3, 0)
                
                # Add quality assessment
                mask_area = np.sum(mask.astype(bool))
                if mask_area > 1000:
                    quality = "EXCELLENT"
                    quality_color = (0, 255, 0)
                elif mask_area > 500:
                    quality = "GOOD"
                    quality_color = (0, 255, 255)
                elif mask_area > 100:
                    quality = "OK"
                    quality_color = (0, 165, 255)
                else:
                    quality = "TOO SMALL"
                    quality_color = (0, 0, 255)
                
                title = f"Preview: {get_object_name(current_obj_id)} - {quality} ({mask_area} pixels)"
                cv2.putText(preview, title, (10, height - 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
                cv2.putText(preview, "Yellow = predicted mask | Close to continue", (10, height - 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.namedWindow('CONSERVATIVE Mask Preview', cv2.WINDOW_NORMAL)
                cv2.imshow('CONSERVATIVE Mask Preview', preview)
                cv2.waitKey(0)
                cv2.destroyWindow('CONSERVATIVE Mask Preview')
                
                print(f"Mask preview: {quality} quality, {mask_area} pixels")
                
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
            
            pos_count = sum(1 for l in labels_dict[current_obj_id] if l == 1)
            
            action = "positive" if label == 1 else "negative"
            guidance = ""
            if label == 1 and pos_count < 8:
                guidance = f" (Add {8-pos_count} more for robust tracking)"
            elif label == 1 and pos_count >= 8:
                guidance = " (Excellent - good for robust tracking!)"
                
            print(f"Added {action} point for {obj_name}{guidance}")
    
    # Initialize display
    img_display = redraw_all_points()
    window_title = 'CONSERVATIVE SAM2 - Emergency Correction' if correction_mode else 'CONSERVATIVE SAM2 - Robust Initial Annotation'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, click_handler)
    
    # Enhanced instructions based on mode
    if correction_mode:
        print("\n🔧 EMERGENCY CORRECTION MODE - Only for severe problems")
        print("=" * 60)
        if current_masks:
            detected_objects = [(obj_id, name) for obj_id, name in object_names.items() 
                            if obj_id in current_masks and np.sum(current_masks[obj_id].astype(bool)) > 0]
            missing_objects = [(obj_id, name) for obj_id, name in object_names.items() 
                            if obj_id not in current_masks or np.sum(current_masks[obj_id].astype(bool)) == 0]
            
            print(f"✅ STILL DETECTED ({len(detected_objects)}): {', '.join([name for _, name in detected_objects])}")
            print(f"❌ MISSING ({len(missing_objects)}): {', '.join([name for _, name in missing_objects])}")
            print("=" * 60)
        
        print("🚨 EMERGENCY: Add points only for completely lost objects")
        print("⚡ Use 8+ positive points per object for robust recovery")
    else:
        print("\n🚀 CONSERVATIVE ANNOTATION MODE - Robust Initial Tracking")
        print("=" * 60)
        print("🎯 GOAL: Create robust initial masks that propagate well")
        print("💪 STRATEGY: More points = better tracking throughout video")
        print("=" * 60)
        print("📋 RECOMMENDED WORKFLOW:")
        print("  1. Add 8-12 positive points per object (spread around edges)")
        print("  2. Add 2-4 negative points if needed (around object boundary)")
        print("  3. Press 'T' to test mask quality")
        print("  4. Name objects with 'C' (use 'target_X' for rings)")
        print("  5. Move to next object with 'N'")
        print("🎯 TIP: More points now = fewer corrections later!")
    
    print("\nControls:")
    print("- Left click: add positive point (ADD MANY!)")
    print("- Right click: add negative point (use sparingly)")
    print("- 'n' for next object, 'p' for previous")
    print("- 'c' to name objects, 't' to test mask")
    print("- Enter to finish, 'q' to quit")
    
    while True:
        cv2.imshow(window_title, img_display)
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
            # Validate point counts for non-correction mode
            if not correction_mode:
                insufficient_objects = []
                for obj_id in points_dict:
                    pos_count = sum(1 for l in labels_dict[obj_id] if l == 1)
                    if pos_count < 5:  # Minimum for decent tracking
                        obj_name = get_object_name(obj_id)
                        insufficient_objects.append(f"{obj_name} ({pos_count} points)")
                
                if insufficient_objects:
                    import tkinter as tk
                    from tkinter import messagebox
                    
                    root = tk.Tk()
                    root.withdraw()
                    
                    proceed = messagebox.askyesno("Low Point Count Warning", 
                        f"These objects have fewer than 5 points:\n" + 
                        "\n".join(insufficient_objects) + 
                        f"\n\nFor robust tracking, 8+ points per object are recommended.\n\n"
                        f"Proceed anyway? (May need more corrections later)")
                    
                    root.destroy()
                    
                    if not proceed:
                        print("Continue adding more points for robust tracking...")
                        continue
            
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Print summary
            total_pos = sum(sum(1 for l in labels if l == 1) for labels in labels_dict.values())
            total_neg = sum(sum(1 for l in labels if l == 0) for labels in labels_dict.values())
            print(f"\n📊 CONSERVATIVE annotation complete:")
            print(f"   Objects: {len(points_dict)}")
            print(f"   Total points: {total_pos} positive, {total_neg} negative")
            print(f"   Average per object: {total_pos/max(len(points_dict), 1):.1f} positive points")
            
            return points_dict, labels_dict, object_names if points_dict else (None, None, None)
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, None, None
    
    return points_dict, labels_dict, object_names

def run_sam2_training(config_path, video_folder, status_callback=None):
    """Real lightweight training using DeepLabV3+"""
    try:
        if status_callback:
            status_callback("Loading training configuration...")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("DeepLabV3+ Fine-tuning Training Started")
        print("=" * 40)
        print(f"Config: {config_path}")
        print(f"Objects: {config['dataset']['object_names']}")
        print(f"Datasets: {len(config['dataset']['annotation_dirs'])}")
        
        if status_callback:
            status_callback("Checking PyTorch availability...")
        
        if not PYTORCH_TRAINING_AVAILABLE:
            error_msg = "❌ PyTorch training components not available. Please install: pip install torch torchvision"
            print(error_msg)
            if status_callback:
                status_callback("Error: PyTorch training not available")
            return False, error_msg
        
        print("✅ PyTorch training components available")
        
        device = setup_device()
        print(f"Using device: {device}")
        
        if status_callback:
            status_callback("Creating dataset...")
        
        dataset = LightweightSegmentationDataset(
            video_folder=video_folder,
            annotation_dirs=config['dataset']['annotation_dirs'],
            object_names=config['dataset']['object_names'],
            image_size=tuple(config['dataset']['image_size'])
        )
        
        if len(dataset) == 0:
            raise ValueError("No training data found!")
        
        print(f"Created dataset with {len(dataset)} samples")
        
        train_size = int(config['dataset']['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        if SKLEARN_AVAILABLE:
            indices = list(range(len(dataset)))
            train_indices, val_indices = train_test_split(
                indices, train_size=train_size, random_state=42
            )
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        
        batch_size = int(config['training']['batch_size'])
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
        
        num_classes = len(config['dataset']['object_names']) + 1
        
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=21)
        except ImportError:
            model = deeplabv3_resnet50(pretrained=True, num_classes=21)
        
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model = model.to(device)
        
        print(f"Model initialized for {num_classes} classes (including background)")
        
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
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
        
        output_dir = os.path.join(video_folder, config['output']['save_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training settings:")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Weight decay: {weight_decay}")
        print(f"  • Device: {device}")
        
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        
        save_every = int(config['output']['save_every'])
        eval_every = int(config['output']['eval_every'])
        
        for epoch in range(num_epochs):
            if status_callback:
                status_callback(f"Training epoch {epoch+1}/{num_epochs}...")
            
            model.train()
            train_loss = 0.0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
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
            
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            images_dict = {img['id']: img for img in coco_data['images']}
            
            image_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
            
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
            
            category_name = ann.get('object_name', 'Unknown')
            if category_name in self.object_names:
                class_id = self.object_names.index(category_name) + 1
            else:
                class_id = 1
            
            try:
                segmentation = ann['segmentation'][0]
                if len(segmentation) >= 6:
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
        
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        mask = self._create_mask_from_annotations(
            sample['annotations'], height, width
        )
        
        image = self.transform(image)
        mask = torch.from_numpy(mask).long()
        
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), 
            size=self.image_size, 
            mode='nearest'
        ).squeeze().long()
        
        return image, mask

def create_finetuning_setup(video_folder):
    """Create fine-tuning setup files in the video folder with auto-detection"""
    try:
        annotation_dirs = []
        object_names_set = set()
        
        print(f"Scanning {video_folder} for annotation files...")
        
        for item in os.listdir(video_folder):
            item_path = os.path.join(video_folder, item)
            if os.path.isdir(item_path) and item.endswith('_frames'):
                coco_file = os.path.join(item_path, "segmentation_coco.json")
                csv_file = os.path.join(item_path, "time_series_metrics.csv")
                
                if os.path.exists(coco_file) and os.path.exists(csv_file):
                    annotation_dirs.append(item_path)
                    print(f"  Found annotations: {item}")
                    
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
        
        annotation_dirs_relative = [os.path.relpath(d, video_folder) for d in annotation_dirs]
        object_names_list = list(object_names_set) if object_names_set else ["Object_1", "Object_2"]
        
        print(f"Found {len(annotation_dirs)} annotated video folders")
        print(f"Detected object names: {object_names_list}")
        
        config_content = f"""# DeepLabV3+ Fine-tuning Configuration - Auto-generated for CONSERVATIVE SAM2
model:
  type: "deeplabv3_resnet50"  # Lightweight segmentation model
  pretrained: true  # Use ImageNet pretrained weights
  
dataset:
  name: "conservative_objects"
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
  save_dir: "conservative_sam2_finetuned"  # Will be created in this folder
  save_every: 5
  eval_every: 2

# Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Base folder: {video_folder}
# Found {len(annotation_dirs)} training datasets
# Model: DeepLabV3+ ResNet50 (lightweight segmentation)
# Training approach: Fine-tune on CONSERVATIVE SAM2 annotations
# Focus: Robust tracking with minimal interference
"""

        config_path = os.path.join(video_folder, "conservative_training_config.yaml")
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
        self.root.title("CONSERVATIVE SAM2 Video Analysis - Minimal Interference")
        self.root.geometry("700x900")
        self.root.minsize(700, 900)
        
        self.device = setup_device()
        self.predictor = None
        self.init_sam2()
        
        self.setup_gui()
        
    def init_sam2(self):
        """Initialize SAM2 predictor"""
        try:
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
        """Setup the GUI with CONSERVATIVE approach messaging"""
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="CONSERVATIVE SAM2 Video Analysis - Minimal Interference", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Conservative approach info
        info_frame = tk.LabelFrame(main_frame, text="🚀 CONSERVATIVE APPROACH", font=("Arial", 9, "bold"))
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = tk.Label(info_frame, 
                            text="✅ Let SAM2 work naturally with minimal interference\n"
                                 "✅ Robust initial annotation (8+ points per object)\n"
                                 "✅ Proper mask history tracking for recovery\n"
                                 "✅ Emergency correction only when ALL objects lost\n"
                                 "✅ Fast processing with fewer interruptions",
                            font=("Arial", 8), fg="darkgreen", justify=tk.LEFT)
        info_text.pack(anchor=tk.W, padx=5, pady=5)
        
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
        options_frame = tk.LabelFrame(main_frame, text="CONSERVATIVE Processing Options", font=("Arial", 9, "bold"))
        options_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Chunk size setting
        chunk_frame = tk.Frame(options_frame)
        chunk_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(chunk_frame, text="Chunk Size (frames):").pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value="500")
        chunk_spin = tk.Spinbox(chunk_frame, from_=100, to=2000, increment=100, 
                               textvariable=self.chunk_size_var, width=10)
        chunk_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(chunk_frame, text="(larger = faster, more memory)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Interactive correction option - CONSERVATIVE DESCRIPTION
        self.interactive_correction = tk.BooleanVar(value=True)
        correction_cb = tk.Checkbutton(options_frame, 
                                      text="🚨 Emergency correction (only when ALL objects lost)",
                                      variable=self.interactive_correction)
        correction_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Analysis video option
        self.create_analysis_video = tk.BooleanVar(value=True)
        analysis_cb = tk.Checkbutton(options_frame, 
                                    text="📊 Create analysis video with plots",
                                    variable=self.create_analysis_video)
        analysis_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Target inclusion detection section
        overlap_frame = tk.LabelFrame(main_frame, text="Ring Inclusion Detection", font=("Arial", 9, "bold"))
        overlap_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Info label
        info_label = tk.Label(overlap_frame, 
                             text='💡 Objects named "target_1", "target_2", etc. will be treated as rings - tracks when objects are INSIDE the ring area',
                             font=("Arial", 8), fg="darkgreen", wraplength=650)
        info_label.pack(anchor=tk.W, padx=5, pady=(5, 2))
        
        # Inclusion threshold setting
        threshold_frame = tk.Frame(overlap_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threshold_frame, text="Inclusion Threshold (%):").pack(side=tk.LEFT)
        self.overlap_threshold_var = tk.StringVar(value="10")
        threshold_spin = tk.Spinbox(threshold_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.overlap_threshold_var, width=10)
        threshold_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(threshold_frame, text="(minimum % of object inside ring to register)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Enable/disable inclusion tracking
        self.enable_overlap_tracking = tk.BooleanVar(value=True)
        overlap_cb = tk.Checkbutton(overlap_frame, 
                                   text="📊 Export ELAN file for ring inclusion analysis",
                                   variable=self.enable_overlap_tracking)
        overlap_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Example text
        example_label = tk.Label(overlap_frame, 
                                text='Example: Ring "target_1" + objects "ball", "hand" → ELAN shows when ball/hand are inside the ring',
                                font=("Arial", 8), fg="blue", wraplength=650)
        example_label.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        # Fine-tuning section
        finetuning_frame = tk.LabelFrame(main_frame, text="🧠 CONSERVATIVE Fine-tuning Workflow", font=("Arial", 9, "bold"))
        finetuning_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Setup button
        setup_button = tk.Button(finetuning_frame, text="🔧 Setup CONSERVATIVE Fine-tuning Environment", 
                 command=self.setup_finetuning, bg="#FF5722", fg="white",
                 font=("Arial", 9))
        setup_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Train button (initially disabled)
        self.train_button = tk.Button(finetuning_frame, text="🚀 Start CONSERVATIVE Training Model", 
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
        
        self.process_button = tk.Button(process_frame, text="🚀 Process Video (CONSERVATIVE Mode)", 
                                       command=self.process_video, bg="#4CAF50", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - CONSERVATIVE mode: Let SAM2 work naturally")
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
            
            self.status_var.set("Setting up CONSERVATIVE fine-tuning environment...")
            self.root.update()
            
            success, num_datasets, object_names, config_path = create_finetuning_setup(folder)
            
            if success:
                self.training_status_var.set(f"CONSERVATIVE setup complete: {num_datasets} datasets, {len(object_names)} object types")
                
                self.train_button.config(state="normal")
                
                messagebox.showinfo("CONSERVATIVE Setup Complete", 
                    f"CONSERVATIVE fine-tuning environment setup complete!\n\n"
                    f"📊 Found {num_datasets} annotated video folders\n"
                    f"🎯 Detected objects: {', '.join(object_names)}\n"
                    f"📝 Config saved: {os.path.basename(config_path)}\n\n"
                    f"✅ Ready to start CONSERVATIVE training!")
                
                self.status_var.set("CONSERVATIVE fine-tuning setup complete - ready to train!")
            else:
                messagebox.showerror("Setup Failed", 
                    "CONSERVATIVE fine-tuning setup failed. Make sure you have processed some videos first.")
                self.status_var.set("CONSERVATIVE fine-tuning setup failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup CONSERVATIVE fine-tuning: {str(e)}")
            self.status_var.set("Setup failed")
    
    def start_training(self):
        """Start training fine-tuned model"""
        try:
            folder = self.folder_var.get()
            if not folder:
                messagebox.showwarning("Warning", "Please select a video folder first")
                return
            
            config_path = os.path.join(folder, "conservative_training_config.yaml")
            if not os.path.exists(config_path):
                messagebox.showerror("Error", 
                    "CONSERVATIVE training configuration not found. Please run setup first.")
                return
            
            result = messagebox.askyesno("Start CONSERVATIVE Training", 
                "Start CONSERVATIVE SAM2 fine-tuning training?\n\n"
                "This will:\n"
                "• Use all annotated videos in this folder\n"
                "• Create a specialized model for your objects\n"
                "• Focus on robust tracking with minimal interference\n"
                "• Take some time to complete\n\n"
                "Continue?")
            
            if not result:
                return
            
            self.status_var.set("CONSERVATIVE training in progress...")
            self.train_button.config(state="disabled")
            self.root.update()
            
            def status_update(message):
                self.training_status_var.set(f"CONSERVATIVE: {message}")
                self.root.update()
            
            success, model_path = run_sam2_training(config_path, folder, status_update)
            
            if success:
                self.training_status_var.set("CONSERVATIVE training completed successfully!")
                
                messagebox.showinfo("CONSERVATIVE Training Complete", 
                    f"🎉 CONSERVATIVE fine-tuning completed successfully!\n\n"
                    f"📁 Model saved: {os.path.basename(model_path)}\n"
                    f"🎯 Ready for robust inference on new videos!\n"
                    f"🚀 Optimized for minimal interference tracking!\n\n"
                    f"✅ You can now use the CONSERVATIVE trained model")
                
                self.status_var.set("CONSERVATIVE training completed - model ready for robust inference!")
            else:
                self.training_status_var.set("CONSERVATIVE training failed")
                messagebox.showerror("Training Failed", f"CONSERVATIVE training failed: {model_path}")
                self.status_var.set("CONSERVATIVE training failed")
            
            self.train_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"CONSERVATIVE training failed: {str(e)}")
            self.training_status_var.set("CONSERVATIVE training failed")
            self.train_button.config(state="normal")
            self.status_var.set("CONSERVATIVE training failed")
    
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
            self.status_var.set(f"Found {len(videos)} video(s) - CONSERVATIVE mode ready")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number_with_preview(self, frames_dir, total_frames):
        """Get frame number with preview functionality"""
        suggested_frame = total_frames // 2
        
        while True:
            frame_num = simpledialog.askinteger(
                "CONSERVATIVE Reference Frame Selection",
                f"Select frame for ROBUST object annotation (0-{total_frames-1}):\n\n"
                f"🚀 CONSERVATIVE approach: Choose a frame with CLEAR object visibility\n"
                f"💪 More initial points = better tracking throughout video\n"
                f"📍 Suggested: Frame {suggested_frame} (middle of video)\n\n"
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
                confirm = messagebox.askyesno("Confirm CONSERVATIVE Frame Selection", 
                    f"Use frame {frame_num} as reference frame?\n\n"
                    "CONSERVATIVE approach:\n"
                    "• This frame will be used for ROBUST object annotation\n"
                    "• Processing will propagate both forward and backward\n"
                    "• More points = fewer corrections needed\n"
                    "• SAM2 will work naturally with minimal interference")
                
                if confirm:
                    return frame_num
            else:
                return None
    
    def process_video(self):
        """Process the selected video with CONSERVATIVE approach"""
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
            self.status_var.set("CONSERVATIVE processing starting...")
            self.root.update()
            
            video_stem = Path(video_name).stem
            frames_dir = os.path.join(folder, f"{video_stem}_frames")
            
            self.status_var.set("Extracting frames...")
            self.root.update()
            
            fps, num_frames = video_to_frames(video_path, frames_dir)
            if fps == -1:
                messagebox.showerror("Error", "Failed to extract frames from video")
                return
            
            frame_num = self.get_frame_number_with_preview(frames_dir, num_frames)
            if frame_num is None:
                self.status_var.set("CONSERVATIVE processing cancelled")
                return
            
            try:
                chunk_size = int(self.chunk_size_var.get())
                overlap_threshold = float(self.overlap_threshold_var.get()) / 100.0
            except ValueError:
                chunk_size = 500
                overlap_threshold = 0.1
                self.chunk_size_var.set("500")
                self.overlap_threshold_var.set("10")
            
            self.status_var.set("Initializing CONSERVATIVE processor...")
            self.root.update()
            
            processor = VideoChunkProcessor(
                predictor=self.predictor, 
                video_dir=frames_dir, 
                chunk_size=chunk_size,
                overlap_frames=30,
                interactive_correction=self.interactive_correction.get(),
                seed_frame_idx=frame_num,
                overlap_threshold=overlap_threshold
            )
            
            frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
            if not os.path.exists(frame_path):
                messagebox.showerror("Error", f"Frame {frame_num} not found")
                return
            
            frame = cv2.imread(frame_path)
            
            self.status_var.set("CONSERVATIVE robust annotation mode...")
            self.root.update()
            
            messagebox.showinfo("CONSERVATIVE Point Selection", 
                f"Frame {frame_num} will open for ROBUST annotation.\n\n"
                "🚀 CONSERVATIVE APPROACH:\n"
                "✓ Add 8-12 positive points per object (more = better)\n"
                "✓ Spread points around object edges and interior\n"
                "✓ Use 'T' to test mask quality before finishing\n"
                "✓ Name objects with 'C' (use 'target_X' for rings)\n\n"
                f"📍 Frame {frame_num} is your reference point\n"
                "🔄 Processing goes both directions with minimal interference\n\n"
                "🎯 TARGET TIP: Name crosshairs/targets as 'target_1', 'target_2'\n"
                "   This will automatically track inclusions with other objects!\n\n"
                "🚀 MORE POINTS NOW = FEWER CORRECTIONS LATER!")
            
            # CONSERVATIVE: Pass correction_mode=False for initial annotation
            points_dict, labels_dict, object_names = select_points_opencv(
                frame, processor, 
                current_masks=None,  # No current masks for initial annotation
                object_names={}, 
                current_frame_idx=frame_num,
                correction_mode=False  # This is initial annotation, not correction
            )
            
            if points_dict is None:
                self.status_var.set("CONSERVATIVE processing cancelled")
                return
            
            self.status_var.set(f"🚀 CONSERVATIVE video processing - letting SAM2 work naturally...")
            self.root.update()
            
            processor.object_names = object_names
            
            results = processor.process_video(points_dict, labels_dict)
            
            if results:
                processor.results = results
                processor.object_names = object_names
                
                self.status_var.set("Saving CONSERVATIVE results...")
                self.root.update()
                
                output_path = os.path.join(frames_dir, "conservative_output_masked.mp4")
                
                if self.enable_overlap_tracking.get() and hasattr(processor, 'overlap_tracker'):
                    print("🎯 Saving CONSERVATIVE results with ELAN export for ring inclusions...")
                    processor.save_results_with_elan(
                        results=results,
                        output_path=output_path,
                        fps=fps,
                        show_original=True,
                        alpha=0.5
                    )
                else:
                    print("📄 Saving CONSERVATIVE standard results...")
                    processor.save_results(
                        output_path=output_path,
                        fps=fps,
                        show_original=True,
                        alpha=0.5
                    )
                
                if self.create_analysis_video.get():
                    self.status_var.set("Creating CONSERVATIVE analysis video...")
                    self.root.update()
                    
                    analysis_output = os.path.join(frames_dir, "conservative_analysis_video.mp4")
                    try:
                        processor.create_analysis_video(
                            results=results,
                            output_path=analysis_output,
                            fps=fps,
                            alpha=0.5
                        )
                        analysis_created = True
                        print("✅ CONSERVATIVE enhanced analysis video created successfully")
                        
                    except Exception as e:
                        print(f"⚠️ Enhanced analysis video failed: {e}")
                        try:
                            simple_output = os.path.join(frames_dir, "conservative_simple_analysis_video.mp4")
                            processor.create_simple_analysis_video(
                                results=results,
                                output_path=simple_output,
                                fps=fps,
                                alpha=0.5
                            )
                            analysis_created = True
                            print("✅ CONSERVATIVE simple analysis video created successfully")
                        except Exception as e2:
                            print(f"❌ Simple analysis video also failed: {e2}")
                            analysis_created = False
                else:
                    analysis_created = False
                
                self.status_var.set("CONSERVATIVE processing completed!")
                
                target_info = ""
                if hasattr(processor, 'overlap_tracker') and processor.overlap_tracker.has_targets():
                    summary = processor.overlap_tracker.get_overlap_summary()
                    target_info = f"\n\n🎯 Ring Inclusion Events:\n"
                    for target_name, data in summary.items():
                        target_info += f"  • {target_name}: {data['total_events']} inclusion events, {data['total_overlap_frames']} frames\n"
                    target_info += "\n📄 ELAN file: target_overlaps.eaf (ring inclusions)"
                
                named_objects = [name for name in object_names.values()]
                objects_summary = "\n".join([f"  • {name}" for name in named_objects])
                
                success_msg = f"""🎉 CONSERVATIVE Processing Complete!

CONSERVATIVE FEATURES USED:
✅ Robust initial annotation (8+ points per object)
✅ Minimal interference with SAM2's natural flow
✅ Proper mask history tracking for recovery
✅ Emergency correction only when ALL objects lost
✅ Let SAM2 propagate naturally for better performance

Reference Frame: {frame_num}
Chunk Size: {chunk_size} frames
Inclusion Threshold: {overlap_threshold*100:.1f}%
Results saved in: {frames_dir}

📁 Generated Files:
• conservative_output_masked.mp4 - Video with ring inclusion indicators
• segmentation_coco.json - Annotations
• time_series_metrics.csv - Movement data"""

                if analysis_created:
                    if os.path.exists(os.path.join(frames_dir, "conservative_analysis_video.mp4")):
                        success_msg += "\n• conservative_analysis_video.mp4 - Enhanced analysis with inclusion highlighting"
                    elif os.path.exists(os.path.join(frames_dir, "conservative_simple_analysis_video.mp4")):
                        success_msg += "\n• conservative_simple_analysis_video.mp4 - Simple analysis video"

                success_msg += f"""{target_info}

📊 Processed Objects ({len(object_names)}):
{objects_summary}

✅ CONSERVATIVE processing completed successfully!
🚀 Approach: Let SAM2 work naturally with robust initial tracking
🔴 Objects inside ring targets are highlighted with green borders"""
                
                messagebox.showinfo("CONSERVATIVE Success", success_msg)
                
            else:
                messagebox.showerror("Error", "CONSERVATIVE video processing failed")
                self.status_var.set("CONSERVATIVE processing failed")
        
        except Exception as e:
            messagebox.showerror("Error", f"CONSERVATIVE processing failed: {str(e)}")
            self.status_var.set("CONSERVATIVE processing failed")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting CONSERVATIVE SAM2 Video Analysis - Minimal Interference Mode!")
    print("=" * 80)
    
    print("\n🚀 CONSERVATIVE APPROACH FEATURES:")
    print("  ✅ Let SAM2 work naturally with minimal interference")
    print("  ✅ Robust initial annotation (8-12 points per object recommended)")
    print("  ✅ Proper mask history tracking for recovery")
    print("  ✅ Emergency correction only when ALL objects are lost")
    print("  ✅ Check interval: Every 1000 frames (vs 100 in aggressive mode)")
    print("  ✅ Much faster processing with fewer interruptions")
    print()
    print("🎯 CORE PHILOSOPHY:")
    print("  • SAM2 is excellent at propagation when given good initial masks")
    print("  • More robust initial points = better propagation throughout video")
    print("  • Corrections should be rare emergency interventions, not frequent tuning")
    print("  • Let the algorithm work - minimal human interference")
    print()
    print("📋 WORKFLOW DIFFERENCES:")
    print("  Old approach: Frequent checks, aggressive correction triggering")
    print("  CONSERVATIVE: Robust initial setup, trust SAM2, minimal interference")
    print()
    print("🔧 CORRECTION BEHAVIOR:")
    print("  • Triggers only when 100% of objects missing for 3+ consecutive frames")
    print("  • Focuses on emergency recovery, not fine-tuning")
    print("  • Proper previous mask detection for restoration")
    print("  • Manual point correction as last resort")
    print()
    print("⚡ PERFORMANCE BENEFITS:")
    print("  • Faster processing (fewer interruptions)")
    print("  • Better tracking quality (SAM2 works naturally)")
    print("  • Fewer user interventions needed")
    print("  • More reliable results")
    print()
    print("🎬 Starting CONSERVATIVE SAM2 application...")
    
    app = VideoAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()