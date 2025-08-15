import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
import torch

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    required_keys = [
        "yolo_conf_threshold", "instance_model_path", "max_history",
        "max_prediction_distance", "max_prediction_frames"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


class ObjectDetector:
    def __init__(self, config_path="config.json"):
        config = load_config(config_path)
        self.config = config
        self.confidence_threshold = config["yolo_conf_threshold"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = None
        self.model_path = config["instance_seg_model_path"]
        self.grid_size = config.get("grid_size", 50)
        self._init_yolo()

        # Motion prediction tracking
        self.previous_detections = []
        self.detection_history = []
        self.max_history = config["max_history"]
        self.max_prediction_distance = config["max_prediction_distance"]
        self.max_prediction_frames = config["max_prediction_frames"]
        self.missing_frames = {}

        self.frame_count = 0
        self.tracking_stats = {
            'total_detections': 0,
            'frames_with_detections': 0,
            'predicted_detections': 0
        }
    
    def _init_yolo(self):
        try:
            self.yolo_model = YOLO(self.model_path)
            self.yolo_model.to(self.device)
        except Exception as e:
            print(f"Failed to initialize YOLO: {e}")
            raise
    
    def detect_objects_frame(self, frame):
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)

            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        if confidence >= self.confidence_threshold:
                            detections.append([float(x1), float(y1), float(x2), float(y2),
                                             float(confidence), int(class_id)])

            return detections

        except Exception as e:
            print(f"Detection error in frame: {e}")
            return []
    
    def smooth_detections(self, current_detections, frame_idx):
        self.frame_count += 1
        self.tracking_stats['total_detections'] += len(current_detections)

        if len(current_detections) > 0:
            self.tracking_stats['frames_with_detections'] += 1

        smoothed_detections = current_detections.copy()

        # Try to predict missing objects
        if len(current_detections) == 0 and len(self.detection_history) >= 2:
            predicted = self._predict_missing_objects()
            smoothed_detections.extend(predicted)
            self.tracking_stats['predicted_detections'] += len(predicted)
        elif len(current_detections) > 0 and len(self.previous_detections) > len(current_detections):
            missing_predicted = self._predict_missing_from_partial()
            smoothed_detections.extend(missing_predicted)
            self.tracking_stats['predicted_detections'] += len(missing_predicted)

        # Update detection history
        self.detection_history.append(current_detections)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # Update missing frame counts
        current_object_ids = set()
        for detection in smoothed_detections:
            obj_id = self._get_object_id(detection)
            current_object_ids.add(obj_id)
            if obj_id in self.missing_frames:
                self.missing_frames[obj_id] = 0

        # Clean up old missing frame entries
        old_ids = [obj_id for obj_id in self.missing_frames.keys()
                   if self.missing_frames[obj_id] > self.max_prediction_frames * 2]
        for obj_id in old_ids:
            del self.missing_frames[obj_id]

        self.previous_detections = smoothed_detections
        return smoothed_detections
    
    def _predict_missing_objects(self):
        predicted = []
        
        if len(self.detection_history) < 2:
            return predicted
        
        recent_objects = self._find_consistent_objects()
        
        for obj_info in recent_objects:
            frames_missing = obj_info['frames_missing']
            if frames_missing <= self.max_prediction_frames:
                predicted_obj = self._predict_object_position(obj_info, frames_missing)
                if predicted_obj:
                    predicted.append(predicted_obj)
        
        return predicted
    
    def _predict_missing_from_partial(self):
        predicted = []

        if len(self.previous_detections) == 0 or len(self.detection_history) < 2:
            return predicted

        current_centers = [self._get_center(det) for det in self.previous_detections]

        for i, prev_det in enumerate(self.previous_detections):
            has_match = False
            prev_center = self._get_center(prev_det)

            # Check if this object is still detected
            for current_det in self.previous_detections:
                current_center = self._get_center(current_det)
                distance = self._calculate_center_distance(prev_center, current_center)
                if distance < self.max_prediction_distance * 0.5:
                    has_match = True
                    break

            # If no match found, try to predict
            if not has_match and len(self.detection_history) >= 2:
                frames_missing = 1
                obj_info = {
                    'detection': prev_det,
                    'frames_missing': frames_missing,
                    'velocity': self._calculate_velocity_for_detection(prev_det)
                }
                predicted_obj = self._predict_object_position(obj_info, frames_missing)
                if predicted_obj:
                    predicted.append(predicted_obj)

        return predicted
    
    def _find_consistent_objects(self):
        consistent_objects = []
        
        if len(self.detection_history) == 0:
            return consistent_objects
        
        last_detections = self.detection_history[-1]
        
        for detection in last_detections:
            obj_id = self._get_object_id(detection)
            
            if obj_id not in self.missing_frames:
                self.missing_frames[obj_id] = 0
            
            self.missing_frames[obj_id] += 1
            
            if self.missing_frames[obj_id] <= self.max_prediction_frames:
                velocity = self._calculate_velocity_for_detection(detection)
                consistent_objects.append({
                    'detection': detection,
                    'frames_missing': self.missing_frames[obj_id],
                    'velocity': velocity,
                    'id': obj_id
                })
        
        return consistent_objects
    
    def _predict_object_position(self, obj_info, frames_missing):
        detection = obj_info['detection']
        velocity = obj_info['velocity']
        
        if velocity is None:
            return None
        
        dx, dy = velocity
        
        # Predict position after missing frames
        pred_x1 = detection[0] + dx * frames_missing
        pred_y1 = detection[1] + dy * frames_missing
        pred_x2 = detection[2] + dx * frames_missing
        pred_y2 = detection[3] + dy * frames_missing
        
        # Reduce confidence based on missing frames
        confidence_reduction = 0.1 * frames_missing
        pred_confidence = max(detection[4] - confidence_reduction, 0.2)
        
        return [pred_x1, pred_y1, pred_x2, pred_y2, pred_confidence, detection[5]]
    
    def _calculate_velocity_for_detection(self, detection):
        if len(self.detection_history) < 2:
            return None
        
        # Find the best match in the previous frame
        prev_frame = self.detection_history[-2]
        best_match = None
        min_distance = float('inf')
        
        for prev_det in prev_frame:
            distance = self._calculate_distance(detection, prev_det)
            if distance < min_distance and distance < self.max_prediction_distance:
                min_distance = distance
                best_match = prev_det
        
        if best_match is None:
            return None
        
        # Calculate velocity
        dx = detection[0] - best_match[0]
        dy = detection[1] - best_match[1]
        
        return (dx, dy)
    
    def _get_object_id(self, detection):
        center_x = int((detection[0] + detection[2]) / 2 / self.grid_size) * self.grid_size
        center_y = int((detection[1] + detection[3]) / 2 / self.grid_size) * self.grid_size
        return f"{center_x}_{center_y}"
    
    def _get_center(self, detection):
        return ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
    
    def _calculate_center_distance(self, center1, center2):
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def _calculate_distance(self, det1, det2):
        center1_x = (det1[0] + det1[2]) / 2
        center1_y = (det1[1] + det1[3]) / 2
        center2_x = (det2[0] + det2[2]) / 2
        center2_y = (det2[1] + det2[3]) / 2
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    def process_video(self, video_path, output_dir):
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        all_detections = {}
        detection_stats = {
            'total_frames': frame_count,
            'frames_with_detections': 0,
            'total_detections': 0,
            'avg_detections_per_frame': 0,
            'confidence_stats': {'min': 1.0, 'max': 0.0, 'avg': 0.0}
        }

        frame_idx = 0
        confidences = []

        with tqdm(total=frame_count, desc="Step 1: Object Detection") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                raw_detections = self.detect_objects_frame(frame)
                smoothed_detections = self.smooth_detections(raw_detections, frame_idx)

                all_detections[frame_idx] = {
                    'detections': smoothed_detections,
                    'frame_shape': (height, width)
                }

                if len(smoothed_detections) > 0:
                    detection_stats['frames_with_detections'] += 1
                    detection_stats['total_detections'] += len(smoothed_detections)

                    for det in smoothed_detections:
                        confidences.append(float(det[4]))

                frame_idx += 1
                pbar.update(1)

        cap.release()

        # Calculate final statistics
        if confidences:
            detection_stats['confidence_stats']['min'] = min(confidences)
            detection_stats['confidence_stats']['max'] = max(confidences)
            detection_stats['confidence_stats']['avg'] = sum(confidences) / len(confidences)

        if detection_stats['frames_with_detections'] > 0:
            detection_stats['avg_detections_per_frame'] = (
                detection_stats['total_detections'] / detection_stats['frames_with_detections']
            )

        # Calculate tracking effectiveness
        tracking_effectiveness = {
            'avg_detections_per_frame': self.tracking_stats['total_detections'] / frame_count if frame_count > 0 else 0,
            'detection_consistency': self.tracking_stats['frames_with_detections'] / frame_count if frame_count > 0 else 0,
            'frames_with_detections': self.tracking_stats['frames_with_detections'],
            'predicted_detections': self.tracking_stats['predicted_detections'],
            'prediction_effectiveness': self.tracking_stats['predicted_detections'] / frame_count if frame_count > 0 else 0
        }

        # Save detection data
        results = {
            'video_info': {
                'path': str(video_path),
                'fps': fps,
                'frame_count': frame_count,
                'resolution': (width, height)
            },
            'detection_stats': detection_stats,
            'tracking_effectiveness': tracking_effectiveness,
            'detections': all_detections,
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'confidence_threshold': self.confidence_threshold,
                'device': self.device,
                'motion_prediction_enabled': True,
                'max_prediction_distance': self.max_prediction_distance,
                'max_history_frames': self.max_history,
                'max_prediction_frames': self.max_prediction_frames,
                'model_path': self.model_path
            }
        }

        # Save results to JSON
        results_file = output_dir / "detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results
