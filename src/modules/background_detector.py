import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import deque
from datetime import datetime

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    required_keys = [
        "max_points", "quality_level", "min_distance", "history_frames", "mask_expansion",
        "blockSize", "useHarrisDetector", "k"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


class BackgroundPointDetector:
    def __init__(self, config_path="config.json"):
        config = load_config(config_path)
        self.config = config
        self.max_points = config["max_points"]
        self.quality_level = config["quality_level"]
        self.min_distance = config["min_distance"]
        self.mask_expansion = config["mask_expansion"]

        self.bbox_history = deque(maxlen=config["history_frames"])

        self.corner_params = dict(
            maxCorners=self.max_points,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=config["blockSize"],
            useHarrisDetector=config["useHarrisDetector"],
            k=config["k"]
        )
        
    def create_object_mask(self, frame_shape, current_detections):
        height, width = frame_shape
        mask = np.ones((height, width), dtype=np.uint8) * 255

        self.bbox_history.append(current_detections.copy())

        # Create exclusion mask using all detections from history
        all_detections = []
        for frame_detections in self.bbox_history:
            all_detections.extend(frame_detections)

        # Mask out all detection areas from history
        for detection in all_detections:
            x1, y1, x2, y2 = map(int, detection[:4])

            # Expand the bounding box
            x1 = max(0, x1 - self.mask_expansion)
            y1 = max(0, y1 - self.mask_expansion)
            x2 = min(width, x2 + self.mask_expansion)
            y2 = min(height, y2 + self.mask_expansion)

            mask[y1:y2, x1:x2] = 0

        return mask
    
    def detect_background_points(self, frame, object_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(
            gray,
            mask=object_mask,
            **self.corner_params
        )
        
        if corners is not None:
            points = corners.reshape(-1, 2)
            return points
        else:
            return np.array([])
    
    def process_video(self, video_path, detection_results_path, output_dir):
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Load detection results
        with open(detection_results_path, 'r') as f:
            detection_data = json.load(f)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        all_background_points = {}
        background_stats = {
            'total_frames': frame_count,
            'frames_with_points': 0,
            'total_points': 0,
            'avg_points_per_frame': 0,
            'point_density_stats': {'min': float('inf'), 'max': 0, 'avg': 0}
        }
        
        frame_idx = 0
        point_counts = []
        
        # Process each frame
        with tqdm(total=frame_count, desc="Step 2: Background Points") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get detections for this frame
                frame_detections = []
                if str(frame_idx) in detection_data['detections']:
                    frame_data = detection_data['detections'][str(frame_idx)]
                    # Try both old and new key names for compatibility
                    if 'detections' in frame_data:
                        frame_detections = frame_data['detections']
                    elif 'tracked_detections' in frame_data:
                        frame_detections = frame_data['tracked_detections']
                    elif 'smoothed_detections' in frame_data:
                        frame_detections = frame_data['smoothed_detections']
                
                # Create object mask
                object_mask = self.create_object_mask((height, width), frame_detections)
                
                # Detect background points
                background_points = self.detect_background_points(frame, object_mask)
                
                # Store results
                all_background_points[frame_idx] = {
                    'points': background_points.tolist() if len(background_points) > 0 else [],
                    'num_points': len(background_points),
                    'object_detections': frame_detections,
                    'frame_shape': (height, width)
                }
                
                # Update statistics
                if len(background_points) > 0:
                    background_stats['frames_with_points'] += 1
                    background_stats['total_points'] += len(background_points)
                    point_counts.append(len(background_points))
                    
                    background_stats['point_density_stats']['min'] = min(
                        background_stats['point_density_stats']['min'], 
                        len(background_points)
                    )
                    background_stats['point_density_stats']['max'] = max(
                        background_stats['point_density_stats']['max'], 
                        len(background_points)
                    )
                
                # Create visualization
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Calculate final statistics
        if point_counts:
            background_stats['point_density_stats']['avg'] = sum(point_counts) / len(point_counts)
            if background_stats['point_density_stats']['min'] == float('inf'):
                background_stats['point_density_stats']['min'] = 0
        else:
            background_stats['point_density_stats']['min'] = 0
        
        if background_stats['frames_with_points'] > 0:
            background_stats['avg_points_per_frame'] = (
                background_stats['total_points'] / background_stats['frames_with_points']
            )
        
        # Save background point data
        results = {
            'video_info': {
                'path': str(video_path),
                'fps': fps,
                'frame_count': frame_count,
                'resolution': (width, height)
            },
            'background_stats': background_stats,
            'background_points': all_background_points,
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'max_points': self.max_points,
                'quality_level': self.quality_level,
                'min_distance': self.min_distance,
                'detection_source': str(detection_results_path)
            }
        }
        
        # Save results to JSON
        results_file = output_dir / "background_points_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
