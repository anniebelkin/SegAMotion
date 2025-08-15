import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue


class VideoVisualizer:
    """Clean, efficient async video visualizer"""
    
    def __init__(self):
        self.video_path = None
        self.output_dir = None
        self.viz_dir = None
        
        # Async processing
        self.task_queue = queue.Queue()
        self.worker = threading.Thread(target=self._process_tasks, daemon=True)
        self.running = True
        self.worker.start()
        
        # Step configurations - all the logic in one place
        self.steps = {
            1: {"file": "detection_results.json", "key": "detections", "color": (0, 255, 0)},
            2: {"file": "background_points_results.json", "key": "background_points", "color": (255, 255, 0)},
            3: {"file": "moving_points_results.json", "key": "frames", "color": (0, 255, 0)},
            4: {"file": "object_tracking_results.json", "key": "frames", "color": (255, 0, 255)},
            5: {"file": "strong_tracking_results.json", "key": "frames", "color": (0, 255, 255)},
        }
    
    def set_video_context(self, video_path: str, output_dir: str):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
    
    def create_step_visualization(self, step: int, mask_data: Dict = None):
        if step == 6:
            self.task_queue.put(("mask_viz", mask_data))
        elif step in self.steps:
            self.task_queue.put(("step_viz", step))
    
    def create_comparison_video(self, bg_path: str, fg_path: str, output_path: str, 
                                bg_title: str = "Background", fg_title: str = "Foreground"):
        """Queue a side-by-side comparison video for async processing"""
        self.task_queue.put(("compare_viz", (bg_path, fg_path, output_path, bg_title, fg_title)))
    
    def wait_for_completion(self):
        self.task_queue.join()
    
    def _process_tasks(self):
        while self.running:
            try:
                task_type, data = self.task_queue.get(timeout=1.0)
                if task_type == "step_viz":
                    self._create_viz(data)
                elif task_type == "mask_viz":
                    self._create_mask_viz(data)
                elif task_type == "compare_viz":
                    self._create_comparison_viz(*data)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except:
                self.task_queue.task_done()
    
    def _create_viz(self, step: int):
        """Universal visualization creator"""
        config = self.steps[step]
        data_file = self.output_dir / config["file"]
        
        if not data_file.exists():
            return
            
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame data and visualize
            frame_data = self._get_frame_data(data, config["key"], frame_idx)
            viz_frame = self._draw_step_data(frame, frame_data, step, frame_idx, config["color"])
            frames.append(viz_frame)
            frame_idx += 1
            
        cap.release()
        
        output_file = self.viz_dir / f"{self.video_path.stem}_step{step}.mp4"
        self._save_video(frames, output_file, fps)
    
    def _create_mask_viz(self, mask_data: Dict):
        """Create mask visualization"""
        if not mask_data:
            return
            
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        masks = mask_data.get('frame_masks', {})
        frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in masks:
                # Apply green overlay
                overlay = np.zeros_like(frame)
                overlay[masks[frame_idx] > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames.append(frame)
            frame_idx += 1
            
        cap.release()
        
        output_file = self.viz_dir / f"{self.video_path.stem}_step6_masking.mp4"
        self._save_video(frames, output_file, fps)
    
    def _get_frame_data(self, data: Dict, key: str, frame_idx: int):
        """Extract frame data from JSON"""
        if key == "detections":
            return data.get(key, {}).get(str(frame_idx), {}).get("detections", [])
        elif key == "background_points":
            return data.get(key, {}).get(str(frame_idx), {}).get("points", [])
        elif key == "frames":
            frame_data = data.get(key, {}).get(str(frame_idx), {})
            # Different steps store data differently
            return frame_data.get("moving_points", frame_data.get("detections", frame_data.get("tracks", [])))
        return []
    
    def _draw_step_data(self, frame: np.ndarray, data: List, step: int, frame_idx: int, color: Tuple):
        """Draw step-specific data on frame"""
        frame = frame.copy()
        
        if step == 1:  # Detections - draw boxes
            for i, det in enumerate(data):
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        elif step in [2, 3]:  # Points - draw circles
            for point in data:
                if len(point) >= 2:
                    x, y = map(int, point[:2])
                    cv2.circle(frame, (x, y), 3, color, -1)
                    
        elif step in [4, 5]:  # Tracks - draw boxes with IDs
            for track in data:
                if isinstance(track, dict) and 'bbox' in track:
                    bbox = track['bbox']
                    track_id = track.get('track_id', track.get('id', 0))
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(frame, f"Step {step} - Frame {frame_idx} - {len(data)} items", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _save_video(self, frames: List[np.ndarray], output_path: Path, fps: float):
        """Save frames as video"""
        if not frames:
            return
            
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    def merge_videos_side_by_side(self, bg_path: str, fg_path: str, output_path: str, 
                                bg_title: str = "Background", fg_title: str = "Foreground"):
        """Merge two videos side by side"""
        bg_cap = cv2.VideoCapture(bg_path)
        fg_cap = cv2.VideoCapture(fg_path)
        if not (bg_cap.isOpened() and fg_cap.isOpened()):
            return
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = bg_cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret_bg, frame_bg = bg_cap.read()
            ret_fg, frame_fg = fg_cap.read()
            if not (ret_bg and ret_fg):
                break
            merged = np.hstack([frame_bg, frame_fg])
            h = merged.shape[0]
            w = merged.shape[1] // 2
            cv2.putText(merged, bg_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(merged, fg_title, (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(merged)
        bg_cap.release()
        fg_cap.release()
        self._save_video(frames, output_path, fps)


# Global instance
visualizer = VideoVisualizer()
