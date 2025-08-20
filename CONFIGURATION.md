# Configuration

## **How to Tune:**

Adjust these parameters in `config.json` to optimize for your video, scene complexity, and desired output. Lower thresholds generally increase sensitivity (more objects, more noise); higher thresholds increase specificity (fewer objects, cleaner results). Model paths let you swap in custom-trained weights for domain adaptation. Visualization options help debug and validate each step.

## Step 1: Object Detection

- **instance_model_path**: Path to YOLO model weights. Use a larger model for higher accuracy, or a smaller one for speed. Custom weights adapt to your domain.
- **yolo_conf_threshold**: Minimum confidence for detections. Increase to reduce false positives (only high-confidence objects detected), decrease to detect more objects (may include noise).
- **max_history**: Number of frames to keep detection history. Higher values smooth out missed detections but may delay response to new objects.

## Step 2: Background Point Detection

- **max_points**: Max background points to track. More points improve motion estimation but increase computation.
- **quality_level**: Minimum quality for feature points. Lower finds more (may include weak/noisy points), higher finds fewer but stronger points.
- **min_distance**: Minimum distance between points. Larger spreads points out, smaller packs them closer (may overlap).
- **blockSize**: Size of block for corner detection. Larger smooths detection, smaller is more sensitive.
- **useHarrisDetector**: Use Harris detector (true) or default (false). Harris is robust but slower.
- **k**: Harris detector parameter. Adjust for sensitivity if using Harris.

## Step 3: Motion Analysis

- **ransac_reproj_threshold**: RANSAC threshold for homography. Lower is stricter (fewer inliers, less noise), higher tolerates more noise (may include outliers).
- **motion_threshold**: Minimum residual motion to consider a point as moving. Lower detects subtle motion (may include background), higher ignores small movements.
- **max_corners**: Max corners for optical flow. More points for accuracy, fewer for speed.
- **lk_winSize**: Window size for optical flow. Larger is robust to noise, smaller is faster.
- **lk_maxLevel**: Pyramid levels for optical flow. More levels handle larger motions.
- **lk_criteria_eps**: Termination epsilon for optical flow. Lower for precision, higher for speed.
- **lk_criteria_count**: Max iterations for optical flow. More for accuracy, fewer for speed.

## Step 4: Adaptive Object Tracking

- **min_points_per_detection**: Minimum motion points inside a detection to consider it dynamic. Lower accepts more objects (may include static ones), higher is stricter (only strong movers).
- **iou_threshold**: IoU threshold for matching detections to tracks. Lower allows looser matches (may cause identity switches), higher is stricter (may miss matches).
- **max_objects**: Max objects to track. More for crowded scenes, fewer for speed.
- **min_motion_duration**: Minimum duration for a track to be valid. Lower accepts short-lived objects, higher requires persistence.
- **min_avg_points_per_frame**: Minimum average motion points per frame for a track. Lower for more tracks, higher for stricter filtering.
- **track_history_length**: Number of frames to keep track history. Higher for stable tracks, lower for responsiveness.

## Step 5: StrongSORT-Inspired Enhanced Tracking

- **max_disappeared**: Max frames a track can be missing before removal. Higher keeps tracks longer (good for occlusions), lower removes quickly (good for fast scenes).
- **min_hits**: Minimum detections before a track is confirmed. Lower confirms quickly (may include false tracks), higher is more reliable (may miss short-lived objects).
- **strong_iou_threshold**: IoU threshold for StrongSORT matching. Lower for looser assignment, higher for stricter.
- **max_tracks**: Max number of tracks. More for crowded scenes, fewer for speed.
- **feature_threshold**: Appearance similarity threshold for matching. Lower for looser matching (may confuse similar objects), higher for stricter (may miss matches).

## Step 6: Masking (Instance Segmentation)

- **instance_seg_model_path**: Path to YOLO segmentation model weights. Use different models for accuracy/speed tradeoff or custom domains.
- **yolo_seg_conf_threshold**: Minimum confidence for segmentation masks. Higher for cleaner masks, lower to include more objects (may include noise).
- **yolo_iou_threshold**: IoU threshold for mask assignment. Adjust for mask quality and overlap.

## Step 7: Video Segmentation (Foreground/Background Separation)

- **use_telea_inpaint**: If true, use TELEA inpainting for background reconstruction (fast, good for small objects). If false, use NS (Navier-Stokes) method (better for large occlusions, slower).
- **background_mask_style**: Style of the background in segmentation output. Options:
  - `black`: Masked background is set to black (default).
  - `blur`: Masked background is blurred.

<img src="gifs/cat-girl_compare.gif" loop=infinite>
