# Pipeline Deep Dive

## Step 1: Object Detection

- **Library**: `ultralytics` YOLO (v8 and custom variants)
- **How it works**: For each frame, the pipeline uses a YOLO model to detect objects and output bounding boxes, class IDs, and confidence scores. The model is loaded once and runs on GPU if available for speed. Detected objects are filtered by confidence threshold (set in config).
- **Detection Smoothing & Prediction**: The pipeline maintains a short history of detections and uses simple motion prediction to handle missed detections. If an object is temporarily missed (e.g., due to occlusion or poor visibility), its position is predicted based on previous frames, reducing flicker and improving temporal consistency.
- **Why YOLO**: YOLO models are fast, accurate, and support both bounding box and segmentation mask outputs. The segmentation head (`-seg` models) allows unified detection and mask prediction, which is crucial for downstream segmentation and tracking. YOLO's flexibility enables use of custom-trained models for specific domains.
- **What it does**: Outputs a list of detected objects per frame, each with bounding box coordinates, confidence, and class. These detections are used to mask out foreground regions, initialize trackers, and guide background point selection.
- **Challenges**: False positives and missed detections can propagate errors to later stages. The pipeline mitigates this by smoothing detections and predicting missing objects, but accuracy still depends on the quality of the YOLO model and input video.
- **Impact**: Accurate object detection is foundational for the entire pipeline. It determines which regions are considered foreground, affects background modeling, and guides all subsequent motion and segmentation analysis.

<img src="gifs/judo_vs_step1.gif" loop=infinite>

## Step 2: Background Point Detection

- **Library**: OpenCV (`goodFeaturesToTrack`)
- **How it works**: For each frame, the pipeline identifies regions not covered by detected objects (using expanded bounding boxes as masks). It then uses OpenCV's corner detection to extract stable feature points from these background regions. These points are tracked across frames to estimate global (camera) motion.
- **Why this method**: Background points are less likely to be affected by moving objects, so they provide a reliable basis for estimating camera motion. By masking out detected objects, the pipeline avoids contaminating background motion estimation with foreground movement.
- **What it does**: Outputs a set of background feature points for each frame. These points are used in the next step to compute optical flow and homography, separating camera motion from object motion.
- **Challenges**: If the background is textureless, occluded, or too small (e.g., crowded scenes), few points may be found, degrading motion estimation. The pipeline uses a history of bounding boxes and adaptive parameters to maximize point quality and distribution.
- **Impact**: Accurate background point selection is critical for robust global motion estimation. Errors here can cause foreground objects to be misclassified or missed entirely in later steps.

## Step 3: Motion Analysis

- **Library**: OpenCV (`calcOpticalFlowPyrLK`, `findHomography`)
- **How it works**: The pipeline tracks the background feature points between consecutive frames using optical flow. It then estimates a global homography (transformation) that best explains the movement of these points, representing camera motion. For each tracked point, the pipeline computes the residual motion - how much its actual movement deviates from the predicted global motion.
- **Why this method**: By modeling and subtracting camera motion, the pipeline isolates independently moving objects. Residuals highlight regions where motion cannot be explained by camera movement alone, which are likely to be foreground objects.
- **What it does**: Outputs per-frame motion vectors and residuals for all tracked points. These are used to cluster and filter moving objects in the next step.
- **Challenges**: Sensitive to background point quality, occlusions, and rapid camera movement. Poor homography estimation can lead to false positives or missed objects. The pipeline uses robust RANSAC fitting and adaptive thresholds to mitigate these issues.
- **Impact**: Accurate motion analysis is essential for distinguishing true object motion from background/camera motion. It enables reliable segmentation of moving objects even in dynamic scenes.

<img src="gifs/step2_vs_step3.gif" loop=infinite>

## Step 4: Adaptive Object Tracking

- **Library**: `scikit-learn` (DBSCAN), NumPy
- **How it works**: This step is responsible for robustly identifying and tracking truly dynamic objects in the scene, even in the presence of noise, occlusions, and background clutter. The process involves several stages:
  1. **Motion Point Clustering**: Residual motion points (from Step 3) are clustered using DBSCAN, which groups points that move together. This helps separate distinct moving objects and filter out background noise.
  2. **Adaptive Filtering**: For each YOLO detection, the pipeline counts how many motion points fall inside its bounding box. An adaptive threshold (based on the distribution of motion points in the frame) is used to decide if a detection is truly dynamic. Only detections with enough motion evidence are kept.
  3. **Track Assignment & Management**: Filtered detections are assigned to existing object tracks using Intersection-over-Union (IoU) matching. Tracks are updated with new detections, and new tracks are created for unmatched detections. Tracks are removed if they are not seen for several frames or if their average motion score drops below a threshold.
  4. **Track Filtering**: After tracking, tracks are filtered by duration and average motion score to ensure only persistent, dynamic objects are kept for downstream segmentation.
- **Why this method**: Combining motion clustering, adaptive filtering, and IoU-based tracking allows the pipeline to handle complex scenes with multiple moving objects, occlusions, and background motion. It ensures that only objects with consistent motion and appearance are tracked, reducing false positives and improving segmentation quality.
- **What it does**: Outputs a set of tracked objects per frame, each with a trajectory, motion statistics, and detection history. These tracks are used for mask generation and final segmentation.
- **Challenges**: Requires careful tuning of clustering and filtering parameters. Complex scenes with overlapping objects, rapid motion, or dense backgrounds can challenge the tracking logic. The pipeline uses adaptive thresholds and track management heuristics to address these issues.
- **Impact**: This step is critical for robust multi-object segmentation and tracking. It enables the pipeline to maintain object identities, handle occlusions, and produce temporally consistent masks for moving objects.

<img src="gifs/step1_vs_step4.gif" loop=infinite>

## Step 5: StrongSORT-Inspired Enhanced Tracking

- **Libraries**: `filterpy` (KalmanFilter), SciPy (Hungarian matching), NumPy
- **How it works**: This step refines and stabilizes object tracks over time using a custom multi-object tracking logic inspired by StrongSORT. Each object track is modeled with a Kalman filter, which predicts its position and velocity in future frames. Detections are assigned to tracks using the Hungarian algorithm (linear assignment), which finds the optimal matching based on Intersection-over-Union (IoU) and simple appearance heuristics (such as color histograms). Tracks are updated with new detections, and new tracks are created for unmatched detections. Tracks are removed if they are not seen for a configurable number of frames or if their motion statistics fall below a threshold.
- **What we actually do**: Unlike the official StrongSORT, our implementation does not use deep appearance features or the full StrongSORT library. Instead, we use Kalman filtering for motion prediction, IoU for assignment, and color-based heuristics for appearance matching. This approach provides robust tracking in the presence of occlusions, missed detections, and appearance changes, but may be less effective in very crowded scenes or with visually similar objects.
- **Why this method**: Our StrongSORT-inspired approach adds memory and inertia to the tracking process, reducing flicker and identity switches. It is computationally efficient and works well for most real-world videos, while remaining flexible for future upgrades.
- **What it does**: Outputs temporally consistent object tracks, each with a unique ID, trajectory, and motion statistics. These tracks are used for mask generation and final segmentation, and can be visualized for analysis.
- **Challenges**: Balancing sensitivity and inertia is key - too much inertia delays reactions to real changes, while too little causes instability and identity switches. Occlusions, rapid motion, and similar-looking objects can still challenge the tracker. The pipeline uses configurable parameters and heuristics to optimize tracking for each video.
- **Impact**: Enhanced tracking is essential for maintaining object identities, handling occlusions, and producing smooth, reliable segmentation masks. It enables the pipeline to track multiple objects robustly across long video sequences.

<img src="gifs/judo_comparison_boomerang.gif" loop=infinite>

## Step 6: Masking (Instance Segmentation)

- **Library**: `ultralytics` YOLO (segmentation models), OpenCV, NumPy
- **How it works**: For each frame with tracked objects, the pipeline runs a YOLO segmentation model to generate instance masks. Masks are filtered and refined by checking if the center of each tracked bounding box (with a threshold relative to its size) falls within the mask. Masks are upscaled, morphologically cleaned, and edge-guided snapping is applied to improve mask boundaries. All valid masks are combined to produce a per-frame foreground mask.
- **Why this method**: Instance segmentation is essential for separating moving objects from the background at the pixel level. Using YOLO segmentation models allows fast, unified detection and mask prediction. The mask filtering logic ensures that only masks corresponding to tracked objects are used, reducing false positives and improving mask quality.
- **What it does**: Outputs a dictionary of per-frame masks for tracked objects, which are used in the next step for background reconstruction and final foreground extraction.
- **Challenges**: Mask quality depends on the YOLO model and input video. Small, overlapping, or fast-moving objects may have coarse or inaccurate masks. The pipeline uses edge-guided snapping and adaptive thresholding to improve results, but further refinement may be needed for challenging scenes.
- **Impact**: Accurate instance masks are critical for high-quality segmentation and background reconstruction. This step enables pixel-level separation of moving objects, supporting downstream analysis and visualization.

<img src="gifs/step5_vs_step6.gif" loop=infinite>

## Step 7: Video Segmentation (Foreground/Background Separation)

- **Library**: OpenCV, NumPy
- **How it works**: For each frame, the pipeline uses the instance masks generated in Step 6 to separate the video into foreground and background. The background is reconstructed using OpenCV's inpainting, which fills masked regions with plausible background pixels. The foreground is extracted by masking out everything except the detected moving objects, resulting in a video where only the segmented objects are visible against a black background.
- **Why this method**: Inpainting provides a fast and effective way to reconstruct backgrounds, especially when objects are small or move slowly. Foreground extraction enables pixel-accurate analysis and visualization of moving objects, supporting downstream tasks like tracking, editing, or further segmentation.
- **What it does**: Outputs two videos per input: one with the reconstructed background (objects removed), and one with the extracted foreground (objects only). Also saves a summary of processing statistics and output file paths.
- **Challenges**: Inpainting can produce artifacts, especially in regions with high texture, motion blur, or large occlusions. Foreground extraction quality depends on mask accuracy. The pipeline uses binary masks and adaptive inpainting methods to improve results, but further refinement may be needed for challenging scenes.
- **Impact**: This step enables clear separation of moving objects from the background, supporting analysis, visualization, and downstream applications such as video editing, object tracking, or generative modeling.

<img src="gifs/judo_compare.gif" loop=infinite>

## Visualization Options

- **Built-in Visualizations**: The pipeline can generate step-by-step visualizations for each major stage, including object detection, background point selection, motion analysis, tracking, masking, and segmentation. These visualizations help users understand and debug the pipeline, and provide clear insight into how each step transforms the video data.
- **How it works**: For each step, annotated videos are created showing detections, tracked points, object tracks, masks, and segmentation results. These are saved in the `visualizations` folder within each output directory. Visualization can be enabled or disabled via the configuration or command-line flags.
- **Compare Option**: As part of the visualization suite, the pipeline supports a "compare" mode, which generates a side-by-side video showing the original input and the segmentation output (foreground/background separation) in sync. This is useful for qualitative evaluation and presentation.
- **Why Visualization Matters**: Visualizations are essential for interpreting results, diagnosing errors, and communicating findings. They make the pipeline accessible to both researchers and practitioners, and support rapid iteration and improvement.
- **Customization**: Visualization options can be customized or extended by modifying the `visualizer.py` module. Users can add new overlays, statistics, or comparison modes as needed for their application.
