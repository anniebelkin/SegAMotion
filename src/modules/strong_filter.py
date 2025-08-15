import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import colorsys
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class Track:
    def __init__(self, track_id, bbox, confidence, class_id, frame_idx):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox
        self.center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.age = 1
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        self.kf = self._create_kalman_filter()
        self.feature_history = []
        self.max_feature_history = 10
        self.history = []
        self.max_history = 30
        
    def _create_kalman_filter(self):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        kf.Q *= 0.1
        kf.R *= 10
        
        x, y = self.center
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)
        
        return kf
    
    def predict(self):
        x, y, w, h = self.kf.x[:4].flatten()
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    
    def advance_prediction(self):
        self.kf.predict()
        self.time_since_update += 1
    
    def update(self, bbox, confidence, frame_idx):
        self.bbox = bbox
        self.confidence = confidence
        self.center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.last_frame = frame_idx
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        x, y = self.center
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.update(np.array([x, y, w, h]).reshape(4, 1))
        
        self.history.append({
            'frame': frame_idx, 'bbox': bbox.copy(),
            'center': self.center.copy(), 'confidence': confidence
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def mark_missed(self):
        self.hit_streak = 0
        self.time_since_update += 1


class StrongSORTTracker:
    def __init__(self, config):
        self.max_disappeared = config["max_disappeared"]
        self.min_hits = config["min_hits"]
        self.iou_threshold = config["strong_iou_threshold"]
        self.feature_threshold = config.get("feature_threshold", 0.6)
        self.max_tracks = config["max_tracks"]
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.colors = {}
    
    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
        return intersection / union if union > 0 else 0.0
    
    def _calculate_feature_similarity(self, bbox1, bbox2, frame):
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        aspect1, aspect2 = w1 / h1 if h1 > 0 else 1.0, w2 / h2 if h2 > 0 else 1.0
        size1, size2 = w1 * h1, w2 * h2
        
        aspect_sim = 1.0 - abs(aspect1 - aspect2) / max(aspect1, aspect2)
        size_sim = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0
        
        return (aspect_sim + size_sim) / 2.0
    
    def _associate_detections_to_tracks(self, detections, frame):
        if len(self.tracks) == 0:
            return [], list(range(len(detections)))
        
        predicted_bboxes = [track.predict() for track in self.tracks]
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou = self._calculate_iou(predicted_bboxes[t], detection['bbox'])
                feature_sim = self._calculate_feature_similarity(
                    predicted_bboxes[t], detection['bbox'], frame
                )
                
                cost = 1.0 - (0.7 * iou + 0.3 * feature_sim)
                cost_matrix[t, d] = cost
        
        matched_tracks, matched_detections = linear_sum_assignment(cost_matrix)
        
        # Filter low-quality matches
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for t, d in zip(matched_tracks, matched_detections):
            if cost_matrix[t, d] < (1.0 - self.iou_threshold):
                matches.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_detections.remove(d)
        
        return matches, unmatched_detections
    
    def update(self, detections, frame):
        self.frame_count += 1
        
        # Advance all track predictions
        for track in self.tracks:
            track.advance_prediction()
        
        # Format detections
        formatted_detections = [{'bbox': det['bbox'], 'confidence': det['confidence'], 
                               'class_id': det['class_id']} for det in detections]
        
        # Associate detections to tracks
        matches, unmatched_detections = self._associate_detections_to_tracks(formatted_detections, frame)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            detection = formatted_detections[det_idx]
            self.tracks[track_idx].update(detection['bbox'], detection['confidence'], self.frame_count)
        
        # Mark unmatched tracks as missed
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in [m[0] for m in matches]]
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks
        for det_idx in unmatched_detections:
            if len(self.tracks) < self.max_tracks:
                detection = formatted_detections[det_idx]
                new_track = Track(self.next_id, detection['bbox'], detection['confidence'],
                                 detection['class_id'], self.frame_count)
                self.tracks.append(new_track)
                
                hue = (self.next_id * 137.508) % 360
                rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
                self.colors[self.next_id] = tuple(int(c * 255) for c in reversed(rgb))
                self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_disappeared]
        return self.get_active_tracks()
    
    def get_active_tracks(self):
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                if track.time_since_update > 0:
                    # Use prediction
                    predicted_bbox = track.predict()
                    predicted_center = np.array([(predicted_bbox[0] + predicted_bbox[2]) / 2,
                                                (predicted_bbox[1] + predicted_bbox[3]) / 2])
                    
                    active_tracks.append({
                        'track_id': track.track_id, 'bbox': predicted_bbox,
                        'confidence': track.confidence * 0.9, 'class_id': track.class_id,
                        'center': predicted_center, 'age': track.age, 'hits': track.hits,
                        'predicted': True, 'time_since_update': track.time_since_update
                    })
                else:
                    # Use actual detection
                    active_tracks.append({
                        'track_id': track.track_id, 'bbox': track.bbox,
                        'confidence': track.confidence, 'class_id': track.class_id,
                        'center': track.center, 'age': track.age, 'hits': track.hits,
                        'predicted': False, 'time_since_update': track.time_since_update
                    })
        return active_tracks


class StrongFilter:
    def __init__(self, config_path=None):
        # Load config at the top
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        self.tracker = StrongSORTTracker(config)
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        x1, y1, x2, y2 = *pt1, *pt2
        dash_length = 10
        
        # Draw dashed lines
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def load_step4_results(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames_data = {int(frame_str): frame_data.get('detections', []) 
                      for frame_str, frame_data in data.get('frames', {}).items()}
        
        return frames_data, data.get('video_info', {})
    
    def process_video(self, video_path, step4_json, output_dir, visualize=True):
        video_path, out_dir = Path(video_path), Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        step4_detections, video_info = self.load_step4_results(step4_json)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
            
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = {
            "video_info": video_info,
            "strongsort_params": {
                "max_disappeared": self.tracker.max_disappeared,
                "min_hits": self.tracker.min_hits,
                "iou_threshold": self.tracker.iou_threshold,
                "max_tracks": self.tracker.max_tracks
            },
            "frames": {}, "tracks": {}
        }
        
        # Process frames
        for frame_idx in tqdm(range(n_frames), desc="Step 5: StrongSORT Tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_detections = step4_detections.get(frame_idx, [])
            active_tracks = self.tracker.update(frame_detections, frame)
            
            frame_result = {
                "step4_detections": len(frame_detections),
                "active_tracks": len(active_tracks),
                "tracks": [{
                    "track_id": track['track_id'],
                    "bbox": track['bbox'].tolist() if hasattr(track['bbox'], 'tolist') else track['bbox'],
                    "confidence": track['confidence'], "class_id": track['class_id'],
                    "center": track['center'].tolist() if hasattr(track['center'], 'tolist') else track['center'],
                    "age": track['age'], "hits": track['hits']
                } for track in active_tracks]
            }
            
            results["frames"][str(frame_idx)] = frame_result
        
        cap.release()
        
        # Compile final tracks
        final_tracks = {}
        for track in self.tracker.tracks:
            if track.hits >= self.tracker.min_hits:
                final_tracks[str(track.track_id)] = {
                    "track_id": track.track_id, "class_id": track.class_id,
                    "first_frame": track.first_frame, "last_frame": track.last_frame,
                    "duration": track.last_frame - track.first_frame + 1,
                    "total_hits": track.hits,
                    "avg_confidence": np.mean([h['confidence'] for h in track.history]),
                    "trajectory": [h['center'].tolist() if hasattr(h['center'], 'tolist') 
                                  else h['center'] for h in track.history]
                }
        
        results["tracks"] = final_tracks
        (out_dir / "strong_tracking_results.json").write_text(json.dumps(results, indent=2))
        return results

