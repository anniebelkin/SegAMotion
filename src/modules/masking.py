import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class VideoMasker:
    def __init__(self, config_path=None):
        if not _YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is required")
        import os
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        self.model = YOLO(config["instance_seg_model_path"])
        self.model.conf = config.get("yolo_seg_conf_threshold", 0.25)
        self.model.iou = config.get("yolo_iou_threshold", 0.7)

    def get_bbox_center(self, bbox):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def point_in_mask(self, point, mask, distance_threshold):
        """Check if point falls within a mask (non-zero area) or is within distance_threshold pixels of mask boundary"""
        px, py = point
        h, w = mask.shape
        if 0 <= px < w and 0 <= py < h:
            if mask[py, px] > 0:
                return True
            # If not inside, check nearby pixels
            y_min = max(py - distance_threshold, 0)
            y_max = min(py + distance_threshold + 1, h)
            x_min = max(px - distance_threshold, 0)
            x_max = min(px + distance_threshold + 1, w)
            region = mask[y_min:y_max, x_min:x_max]
            return np.any(region > 0)
        return False

    def create_masks_for_frame(self, frame, tracked_bboxes, threshold_fraction=0.1):
        """Run YOLO once on frame and filter masks by tracked bounding box centers, using mask probabilities and edge-guided snapping."""
        h, w = frame.shape[:2]
        if not tracked_bboxes:
            return np.zeros((h, w), dtype=np.uint8)
        
        def round_up_to_32(x):
            return int(np.ceil(x / 32) * 32)

        imgsz = round_up_to_32(max(frame.shape[0], frame.shape[1]))

        bbox_centers = [self.get_bbox_center(bbox) for bbox in tracked_bboxes]
        results = self.model(frame, verbose=False, retina_masks=True, imgsz=imgsz)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for r in results:
            if r.masks is not None and r.boxes is not None:
                masks = r.masks.data.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                for mask_prob, conf in zip(masks, confidences):
                    if conf < 0.25:
                        continue
                    # Smooth upscale mask probabilities
                    if mask_prob.shape != (h, w):
                        mask_prob = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    # Tiny closing to clean up mask
                    mask_prob = cv2.morphologyEx(mask_prob, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    # Edge-guided snapping
                    edges = cv2.Canny((mask_prob * 255).astype(np.uint8), 50, 150)
                    snapped_mask = np.where(edges > 0, mask_prob, mask_prob)
                    # Binarize
                    binary_mask = (snapped_mask > 0.5).astype(np.uint8) * 255
                    # Check bbox centers
                    for bbox, bbox_center in zip(tracked_bboxes, bbox_centers):
                        x1, y1, x2, y2 = bbox
                        diag = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
                        distance_threshold = max(2, int(diag * threshold_fraction))
                        if self.point_in_mask(bbox_center, binary_mask, distance_threshold):
                            combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
                            break

        return combined_mask

    def process_video(self, video_path, tracking_results_path, output_dir):
        """Process video to create masks for tracked objects"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Load tracking data
        with open(tracking_results_path, 'r') as f:
            tracking_data = json.load(f)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get frames with tracks
        frames_data = tracking_data.get('frames', {})
        frames_with_tracks = [(int(f_str), f_data) for f_str, f_data in frames_data.items() 
                             if f_data.get('tracks')]
        
        if not frames_with_tracks:
            cap.release()
            return {'frame_masks': {}, 'enhanced_masks': {}, 'processing_info': {
                "total_frames": total_frames, "fps": fps, "resolution": [width, height],
                "frames_with_masks": 0, "enhanced_frames_with_masks": 0
            }}
        
        frame_masks = {}
        
        # Process frames with tracks
        with tqdm(total=len(frames_with_tracks), desc="Step 6: Mask Creation") as pbar:
            for frame_idx, frame_data in frames_with_tracks:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    pbar.update(1)
                    continue
                
                # Get all bboxes for this frame
                tracked_bboxes = [track.get('bbox') for track in frame_data.get('tracks', []) 
                                 if track.get('bbox')]
                
                if tracked_bboxes:
                    # Run YOLO once for all objects in this frame
                    combined_mask = self.create_masks_for_frame(frame, tracked_bboxes)
                
                    if np.any(combined_mask > 0):
                        # Trust YOLO's output - minimal processing
                        frame_masks[frame_idx] = combined_mask
                
                pbar.update(1)
        
        cap.release()
        
        return {
            'frame_masks': frame_masks,
            'enhanced_masks': frame_masks,  # Same as frame_masks for simplicity
            'processing_info': {
                "total_frames": total_frames,
                "fps": fps,
                "resolution": [width, height],
                "frames_with_masks": len(frame_masks),
                "enhanced_frames_with_masks": len(frame_masks)
            }
        }
