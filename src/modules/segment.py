import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class VideoSegmenter:
    def __init__(self, config_path=None):
        import os
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        use_telea = config.get("use_telea_inpaint", True)
        if use_telea:
            self.inpaint_method = cv2.INPAINT_TELEA
        else:
            self.inpaint_method = cv2.INPAINT_NS
        self.background_mask_style = config.get("background_mask_style", "black")

    def load_masking_results(self, results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def process_video(self, video_path, masking_data, output_dir):

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Extract mask data (passed directly from masking step)
        frame_masks = masking_data.get('frame_masks', {})
        enhanced_masks = masking_data.get('enhanced_masks', {})
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create visualizations directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output video writers 
        bg_output_path = output_dir / f"{video_path.stem}_background.mp4"
        fg_output_path = output_dir / f"{video_path.stem}_foreground.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        bg_writer = cv2.VideoWriter(str(bg_output_path), fourcc, fps, (width, height))
        fg_writer = cv2.VideoWriter(str(fg_output_path), fourcc, fps, (width, height))
        
        # Get masks from the in-memory data
        masks_data = frame_masks
        enhanced_masks_data = enhanced_masks
        
        # Process each frame
        frame_idx = 0
        frames_with_masks = 0
        with tqdm(total=total_frames, desc="Step 7: Creating background/foreground videos") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Create combined mask for this frame
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                frame_key = frame_idx  # Use integer key, not string
                
                # Use enhanced mask if available, otherwise use regular mask
                mask_available = False
                if frame_key in enhanced_masks_data:
                    mask = enhanced_masks_data[frame_key]
                    if mask is not None and np.any(mask > 0):
                        # Mask is already a numpy array - much faster!
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                        mask_available = True
                elif frame_key in masks_data:
                    mask = masks_data[frame_key]
                    if mask is not None and np.any(mask > 0):
                        # Mask is already a numpy array - much faster!
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                        mask_available = True
                
                if mask_available:
                    frames_with_masks += 1
                
                # Apply inpainting or masking to create background
                if np.any(combined_mask > 0):
                    # Ensure mask is single channel and has correct values (0 or 255)
                    if len(combined_mask.shape) == 3:
                        combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)
                    combined_mask = (combined_mask > 0).astype(np.uint8) * 255

                    if self.background_mask_style == "black":
                        background_frame = frame.copy()
                        background_frame[combined_mask > 0] = 0
                    elif self.background_mask_style == "blur":
                        # Blur masked regions (previous behavior)
                        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
                        background_frame = frame.copy()
                        background_frame[combined_mask > 0] = blurred[combined_mask > 0]
                    else:
                        # Default to black if unknown style
                        background_frame = frame.copy()
                        background_frame[combined_mask > 0] = 0
                else:
                    background_frame = frame.copy()
                
                # Create foreground with black background
                foreground_frame = np.zeros_like(frame)
                if np.any(combined_mask > 0):
                    foreground_frame[combined_mask > 0] = frame[combined_mask > 0]
                
                # Write frames
                bg_writer.write(background_frame)
                fg_writer.write(foreground_frame)
                
                # Update progress bar with more info
                if frame_idx % 10 == 0:  # Update description every 10 frames
                    pbar.set_description(f"Step 7: Processing frame {frame_idx+1}/{total_frames} ({frames_with_masks} with masks)")
                
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        bg_writer.release()
        fg_writer.release()
        
        # Save segmentation results
        results = {
            "video_path": str(video_path),
            "processing_info": {
                "total_frames": total_frames,
                "fps": fps,
                "resolution": [width, height],
                "inpaint_method": "TELEA" if self.inpaint_method == cv2.INPAINT_TELEA else "NS"
            },
            "output_files": {
                "background_video": str(bg_output_path),
                "foreground_video": str(fg_output_path)
            },
            "frames_processed": frame_idx,
            "frames_with_masks": frames_with_masks
        }
        
        results_path = output_dir / "segmentation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        return results
