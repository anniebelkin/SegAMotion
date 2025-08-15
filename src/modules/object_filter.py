import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict
import colorsys

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    required_keys = [
        "max_objects", "track_history_length", "iou_threshold",
        "min_points_per_detection", "min_motion_duration", "min_avg_points_per_frame"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


class ObjectFilter:
    def __init__(self, config_path="config.json"):
        config = load_config(config_path)
        self.config = config
        self.max_objects = config["max_objects"]
        self.track_history_length = config["track_history_length"]
        self.iou_threshold = config["iou_threshold"]
        self.min_points_per_detection = config["min_points_per_detection"]
        self.min_motion_duration = config["min_motion_duration"]
        self.min_avg_points_per_frame = config["min_avg_points_per_frame"]

        self.active_tracks = {}
        self.next_track_id = 1
        self.object_colors = {}
        self.completed_tracks = {}
        
    def load_motion_points(self, json_path):
        data = json.loads(Path(json_path).read_text())
        motion_points = {int(k): np.array(v.get('moving_points', []), dtype=np.float32) 
                        for k, v in data['frames'].items()}
        return motion_points, data['video_info']
        
    def load_yolo_detections(self, json_path):
        data = json.loads(Path(json_path).read_text())
        yolo_detections = {int(k): np.array(v.get('detections', []), dtype=np.float32) 
                          for k, v in data['detections'].items()}
        return yolo_detections, data['video_info']
    def filter_yolo_by_motion_adaptive(self, yolo_detections, motion_points):
        if len(yolo_detections) == 0 or len(motion_points) == 0:
            return []
            
        # Count motion points in each detection
        detection_points = []
        for detection in yolo_detections:
            x1, y1, x2, y2, confidence, class_id = detection
            points_in_bbox = [p for p in motion_points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
            detection_points.append({
                'detection': detection,
                'points_inside': len(points_in_bbox),
                'points_in_bbox': points_in_bbox
            })
        
        max_points_in_frame = max((d['points_inside'] for d in detection_points), default=0)
        adaptive_threshold = max(self.min_points_per_detection, max_points_in_frame / 10.0)
        
        # Filter detections
        filtered_detections = []
        for item in detection_points:
            if item['points_inside'] >= adaptive_threshold:
                x1, y1, x2, y2, confidence, class_id = item['detection']
                motion_score = item['points_inside'] / len(motion_points) if len(motion_points) > 0 else 0
                
                filtered_detections.append({
                    'bbox': np.array([x1, y1, x2, y2]),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'motion_points': np.array(item['points_in_bbox']),
                    'motion_score': motion_score,
                    'num_points_in_bbox': item['points_inside'],
                    'adaptive_threshold_used': adaptive_threshold,
                    'max_points_in_frame': max_points_in_frame
                })
        
        return filtered_detections
        
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
        return intersection / union if union > 0 else 0.0
        
    def assign_detections_to_tracks(self, detections):
        if not self.active_tracks or not detections:
            return {}, list(range(len(detections)))
            
        track_ids = list(self.active_tracks.keys())
        iou_matrix = np.array([[self.calculate_iou(self.active_tracks[track_id]['bbox'], det['bbox'])
                               for det in detections] for track_id in track_ids])
        
        assignments = {}
        unassigned_detections = list(range(len(detections)))
        
        for i, track_id in enumerate(track_ids):
            if not unassigned_detections:
                break
                
            valid_ious = [(j, iou_matrix[i, j]) for j in unassigned_detections 
                         if iou_matrix[i, j] > self.iou_threshold]
            
            if valid_ious:
                best_j, _ = max(valid_ious, key=lambda x: x[1])
                assignments[track_id] = best_j
                unassigned_detections.remove(best_j)
                
        return assignments, unassigned_detections
        
    def _update_track(self, track, detection, frame_idx):
        x1, y1, x2, y2 = detection['bbox']
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        track.update({
            'bbox': detection['bbox'], 'confidence': detection['confidence'],
            'class_id': detection['class_id'], 'motion_points': detection['motion_points'],
            'motion_score': detection['motion_score'], 'last_seen': frame_idx,
            'center': center, 'age': track['age'] + 1,
            'total_points_in_bbox': track['total_points_in_bbox'] + detection['num_points_in_bbox']
        })
        
        track['history'].append({
            'frame': frame_idx, 'bbox': detection['bbox'].copy(),
            'center': center.copy(), 'motion_score': detection['motion_score']
        })
        
        if len(track['history']) > self.track_history_length:
            track['history'].pop(0)
    
    def _create_new_track(self, detection, frame_idx):
        track_id = self.next_track_id
        self.next_track_id += 1
        
        hue = (track_id * 137.508) % 360
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
        self.object_colors[track_id] = tuple(int(c * 255) for c in reversed(rgb))
        
        x1, y1, x2, y2 = detection['bbox']
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        return track_id, {
            'id': track_id, 'bbox': detection['bbox'], 'confidence': detection['confidence'],
            'class_id': detection['class_id'], 'motion_points': detection['motion_points'],
            'motion_score': detection['motion_score'], 'center': center,
            'first_seen': frame_idx, 'last_seen': frame_idx, 'age': 1,
            'total_points_in_bbox': detection['num_points_in_bbox'],
            'history': [{'frame': frame_idx, 'bbox': detection['bbox'].copy(),
                        'center': center.copy(), 'motion_score': detection['motion_score']}]
        }

    def update_tracks(self, detections, frame_idx):
        assignments, unassigned_detections = self.assign_detections_to_tracks(detections)
        
        # Update existing tracks
        for track_id, detection_idx in assignments.items():
            self._update_track(self.active_tracks[track_id], detections[detection_idx], frame_idx)
                
        # Create new tracks
        for detection_idx in unassigned_detections:
            if len(self.active_tracks) >= self.max_objects:
                break
            track_id, track = self._create_new_track(detections[detection_idx], frame_idx)
            self.active_tracks[track_id] = track
            
        # Remove old or low-quality tracks
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if frame_idx - track['last_seen'] > 5:
                self.completed_tracks[track_id] = track
                tracks_to_remove.append(track_id)
            else:
                motion_duration = track['last_seen'] - track['first_seen'] + 1
                avg_points = track.get('total_points_in_bbox', 0) / motion_duration if motion_duration > 0 else 0
                if motion_duration >= 5 and avg_points < self.min_avg_points_per_frame * 0.5:
                    tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            self.active_tracks.pop(track_id, None)
            
    def filter_tracks_by_duration(self):
        return {tid: track for tid, track in self.completed_tracks.items()
                if track['last_seen'] - track['first_seen'] + 1 >= self.min_motion_duration}
            
    def process_video(self, video_path, motion_json, detection_json, output_dir, visualize=True):
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load motion points from Step 3 and YOLO detections from Step 1
        motion_points, motion_video_info = self.load_motion_points(motion_json)
        yolo_detections, yolo_video_info = self.load_yolo_detections(detection_json)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
            
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = {
            "video_info": motion_video_info,
            "segmentation_params": {
                "min_points_per_detection": 3,
                "iou_threshold": self.iou_threshold,
                "max_objects": self.max_objects,
                "min_motion_duration": self.min_motion_duration,
                "min_avg_points_per_frame": self.min_avg_points_per_frame
            },
            "frames": {},
            "tracks": {}
        }
        
        for frame_idx in tqdm(range(n_frames), desc="Step 4: Filtering objects"):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_detections = []
            
            frame_yolo = yolo_detections.get(frame_idx, np.array([]))
            frame_motion = motion_points.get(frame_idx, np.array([]))
            
            if len(frame_yolo) > 0 and len(frame_motion) > 0:
                filtered_detections = self.filter_yolo_by_motion_adaptive(
                    frame_yolo, frame_motion
                )
                frame_detections = filtered_detections
                        
            self.update_tracks(frame_detections, frame_idx)
            
            frame_result = {
                "num_yolo_detections": len(frame_yolo),
                "num_motion_points": len(frame_motion),
                "num_filtered_detections": len(frame_detections),
                "num_tracks": len(self.active_tracks),
                "adaptive_threshold": frame_detections[0]['adaptive_threshold_used'] if frame_detections else None,
                "max_points_in_frame": frame_detections[0]['max_points_in_frame'] if frame_detections else None,
                "detections": []
            }
            
            frame_result["detections"] = [{
                "bbox": det['bbox'].tolist(), "confidence": det['confidence'],
                "class_id": det['class_id'], "motion_score": det['motion_score'],
                "num_points_in_bbox": det['num_points_in_bbox']
            } for det in frame_detections]
                
            results["frames"][str(frame_idx)] = frame_result
                
        cap.release()
        
        # Filter tracks by duration and motion criteria
        all_tracks = {**self.active_tracks, **self.completed_tracks}
        valid_tracks = {}
        
        for track_id, track in all_tracks.items():
            duration = track['last_seen'] - track['first_seen'] + 1
            avg_points = track.get('total_points_in_bbox', 0) / duration if duration > 0 else 0
            
            if duration >= self.min_motion_duration and avg_points >= self.min_avg_points_per_frame:
                valid_tracks[track_id] = track
            
        # Save track summaries
        for track_id, track in valid_tracks.items():
            duration = track['last_seen'] - track['first_seen'] + 1
            avg_points = track.get('total_points_in_bbox', 0) / duration if duration > 0 else 0
            
            results["tracks"][str(track_id)] = {
                "id": track_id, "class_id": track['class_id'],
                "first_frame": track['first_seen'], "last_frame": track['last_seen'],
                "duration": duration, "age": track['age'],
                "total_points_in_bbox": track['total_points_in_bbox'],
                "avg_points_per_frame": avg_points,
                "avg_motion_score": np.mean([h['motion_score'] for h in track['history']]),
                "trajectory": [h['center'].tolist() for h in track['history']]
            }
        (out_dir / "object_tracking_results.json").write_text(json.dumps(results, indent=2))
        return results