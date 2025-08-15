# SegAMotion: Video Segmentation & Tracking

SegAMotion is a fast, modular pipeline for video object segmentation and tracking. It combines deep learning (YOLO) and classical computer vision to separate moving objects from background, even in challenging videos with camera motion, crowds, or occlusions. The pipeline is easy to configure, produces clean outputs, and includes visualizations for instant feedback.

---

<table>
  <tr>
    <td><img src="gifs/rollerblade_compare.gif" loop=infinite></td>
    <td><img src="gifs/parkour_compare.gif" loop=infinite></td>
  </tr>
  <tr>
    <td><img src="gifs/drift-straight_compare.gif" loop=infinite></td>
    <td><img src="gifs/longboard_compare.gif" loop=infinite></td>
  </tr>
</table>

## Quick Start

### Setup Environment

```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Download YOLO models and place in `data/yolo/`:

- [`yolo11m-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt): medium, balanced (default)
- [`yolo11l-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt): large, accurate
- [`yolo11x-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt): x-large, best but slow

### Usage

Run the pipeline on a video:

```bash
python main.py --path <video_path>
```

- `--path <video_path>`: Specify the video to process. The pipeline will look for the file in the given path, the `data/vid/` folder, or the project root. If not provided, all videos in `data/vid/` will be processed.

Show side-by-side comparison of input and segmentation output:

```bash
python main.py --path <video_path> --compare
```

- `--compare`: Generates a side-by-side video showing the original input and the segmentation output for easy visual comparison.

Disable visualizations:

```bash
python main.py --path <video_path> --no-viz
```

- `--no-viz`: Disables creation of step-by-step visualization videos (faster, less disk usage).

Get help and see all options:

```bash
python main.py --help
```

Outputs (videos, logs, visualizations) are saved in the `output/` directory, organized by video name and timestamp.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Step 1: Object Detection](#step-1-object-detection)
  - [Step 2: Background Point Detection](#step-2-background-point-detection)
  - [Step 3: Motion Analysis](#step-3-motion-analysis)
  - [Step 4: Adaptive Object Tracking](#step-4-adaptive-object-tracking)
  - [Step 5: StrongSORT-Inspired Enhanced Tracking](#step-5-strongsort-inspired-enhanced-tracking)
  - [Step 6: Masking (Instance Segmentation)](#step-6-masking-instance-segmentation)
  - [Step 7: Video Segmentation (Foreground/Background Separation)](#step-7-video-segmentation-foregroundbackground-separation)
  - [Compare Mode](#compare-mode)
- [Configuration](#configuration)
- [Dependencies &amp; Libraries](#dependencies--libraries)
- [Limitations &amp; Future Work](#limitations--future-work)

---

## Project Overview

SegAMotion is designed for robust video object segmentation and tracking in real-world scenarios: moving cameras, crowded scenes, occlusions, and complex interactions. It combines YOLO-based deep learning for detection/segmentation with classical computer vision (feature tracking, optical flow, homography, clustering, Kalman filtering) for global motion modeling and adaptive tracking.

**Core Features:**

- End-to-end pipeline: From raw video to pixel-accurate foreground/background separation, object tracks, and visualizations.
- Flexible detection & segmentation: Use any YOLO model for speed or accuracy; custom weights supported.
- Global motion compensation: Removes camera-induced motion to isolate true object movement.
- Adaptive tracking: Maintains object identities, handles occlusions, and filters static objects.
- Instance masking & inpainting: Clean separation of moving objects and background for analysis or editing.
- Configurable: All parameters in a single config file for easy tuning and reproducibility.
- Visual feedback: Step-by-step visualizations and compare mode for instant validation.

**How it works:**
Each module refines and validates outputs from the previous step, minimizing errors and preserving object identities. The modular structure supports easy extension, benchmarking, and integration with other tools.

**Applications:**

- Research, surveillance, robotics, sports analytics, video editing.
- Benchmarking and analysis on datasets like DAVIS 2017, with support for custom videos and models.

## Benchmark & Validation

SegAMotion has been tested on diverse scenes from the DAVIS 2017 dataset. On average, each video required about one minute of processing over CPU, demonstrating practical efficiency for research workflows.

We compared SegAMotion's results to the outputs of Detectron2 and YOLO alone, focusing on the ability to filter out static (non-moving) objects. In most cases, SegAMotion successfully removed static objects that were present in the raw detection outputs, providing cleaner and more relevant segmentation results for dynamic scenes.

The pipeline was evaluated on videos with different types of camera motion and object movement, showing robustness to dynamic backgrounds, occlusions, and challenging scenarios. These results highlight SegAMotion's effectiveness in distinguishing moving objects from static background, even under complex conditions.

<table>
  <tr>
    <td><img src="gifs/bike-packing_full_compare.gif" loop=infinite></td>
    <td><img src="gifs/boxing-fisheye_full_compare.gif" loop=infinite></td>
  </tr>
  <tr>
    <td><img src="gifs/car-roundabout_full_compare.gif" loop=infinite></td>
  <td><img src="gifs/static_background_full_compare.gif" loop=infinite></td>
  </tr>
  <tr>
  <td><img src="gifs/horsejump-high_full_compare.gif" loop=infinite></td>
  <td><img src="gifs/judo_full_compare.gif" loop=infinite></td>
  </tr>
</table>

For more, you can download some saved results from [`HERE`]([https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt](https://drive.google.com/file/d/154ihko5SqyqB4uqCkP6hRf9reFYgIMbB/view?usp=sharing))

---

## Directory Structure

```text
.
├── config.json              # Pipeline configuration
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── data/
│   ├── vid/                 # Input videos
│   └── yolo/                # YOLO model weights
├── src/
│   ├── modules/             # Core pipeline modules
│   │   ├── object_detector.py
│   │   ├── background_detector.py
│   │   ├── motion_analyzer.py
│   │   ├── object_filter.py
│   │   ├── strong_filter.py
│   │   ├── masking.py
│   │   ├── segment.py
│   └── utils/               # Utility functions and visualization
│       ├── visualizer.py
```

---

## Pipeline Deep Dive

### Step 1: Object Detection

- **Library**: `ultralytics` YOLO (v8 and custom variants)
- **How it works**: For each frame, the pipeline uses a YOLO model to detect objects and output bounding boxes, class IDs, and confidence scores. The model is loaded once and runs on GPU if available for speed. Detected objects are filtered by confidence threshold (set in config).
- **Detection Smoothing & Prediction**: The pipeline maintains a short history of detections and uses simple motion prediction to handle missed detections. If an object is temporarily missed (e.g., due to occlusion or poor visibility), its position is predicted based on previous frames, reducing flicker and improving temporal consistency.
- **Why YOLO**: YOLO models are fast, accurate, and support both bounding box and segmentation mask outputs. The segmentation head (`-seg` models) allows unified detection and mask prediction, which is crucial for downstream segmentation and tracking. YOLO's flexibility enables use of custom-trained models for specific domains.
- **What it does**: Outputs a list of detected objects per frame, each with bounding box coordinates, confidence, and class. These detections are used to mask out foreground regions, initialize trackers, and guide background point selection.
- **Challenges**: False positives and missed detections can propagate errors to later stages. The pipeline mitigates this by smoothing detections and predicting missing objects, but accuracy still depends on the quality of the YOLO model and input video.
- **Impact**: Accurate object detection is foundational for the entire pipeline. It determines which regions are considered foreground, affects background modeling, and guides all subsequent motion and segmentation analysis.

---

### Step 2: Background Point Detection

- **Library**: OpenCV (`goodFeaturesToTrack`)
- **How it works**: For each frame, the pipeline identifies regions not covered by detected objects (using expanded bounding boxes as masks). It then uses OpenCV's corner detection to extract stable feature points from these background regions. These points are tracked across frames to estimate global (camera) motion.
- **Why this method**: Background points are less likely to be affected by moving objects, so they provide a reliable basis for estimating camera motion. By masking out detected objects, the pipeline avoids contaminating background motion estimation with foreground movement.
- **What it does**: Outputs a set of background feature points for each frame. These points are used in the next step to compute optical flow and homography, separating camera motion from object motion.
- **Challenges**: If the background is textureless, occluded, or too small (e.g., crowded scenes), few points may be found, degrading motion estimation. The pipeline uses a history of bounding boxes and adaptive parameters to maximize point quality and distribution.
- **Impact**: Accurate background point selection is critical for robust global motion estimation. Errors here can cause foreground objects to be misclassified or missed entirely in later steps.

---

### Step 3: Motion Analysis

- **Library**: OpenCV (`calcOpticalFlowPyrLK`, `findHomography`)
- **How it works**: The pipeline tracks the background feature points between consecutive frames using optical flow. It then estimates a global homography (transformation) that best explains the movement of these points, representing camera motion. For each tracked point, the pipeline computes the residual motion - how much its actual movement deviates from the predicted global motion.
- **Why this method**: By modeling and subtracting camera motion, the pipeline isolates independently moving objects. Residuals highlight regions where motion cannot be explained by camera movement alone, which are likely to be foreground objects.
- **What it does**: Outputs per-frame motion vectors and residuals for all tracked points. These are used to cluster and filter moving objects in the next step.
- **Challenges**: Sensitive to background point quality, occlusions, and rapid camera movement. Poor homography estimation can lead to false positives or missed objects. The pipeline uses robust RANSAC fitting and adaptive thresholds to mitigate these issues.
- **Impact**: Accurate motion analysis is essential for distinguishing true object motion from background/camera motion. It enables reliable segmentation of moving objects even in dynamic scenes.

---

### Step 4: Adaptive Object Tracking

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

---

### Step 5: StrongSORT-Inspired Enhanced Tracking

- **Libraries**: `filterpy` (KalmanFilter), SciPy (Hungarian matching), NumPy
- **How it works**: This step refines and stabilizes object tracks over time using a custom multi-object tracking logic inspired by StrongSORT. Each object track is modeled with a Kalman filter, which predicts its position and velocity in future frames. Detections are assigned to tracks using the Hungarian algorithm (linear assignment), which finds the optimal matching based on Intersection-over-Union (IoU) and simple appearance heuristics (such as color histograms). Tracks are updated with new detections, and new tracks are created for unmatched detections. Tracks are removed if they are not seen for a configurable number of frames or if their motion statistics fall below a threshold.
- **What we actually do**: Unlike the official StrongSORT, our implementation does not use deep appearance features or the full StrongSORT library. Instead, we use Kalman filtering for motion prediction, IoU for assignment, and color-based heuristics for appearance matching. This approach provides robust tracking in the presence of occlusions, missed detections, and appearance changes, but may be less effective in very crowded scenes or with visually similar objects.
- **Why this method**: Our StrongSORT-inspired approach adds memory and inertia to the tracking process, reducing flicker and identity switches. It is computationally efficient and works well for most real-world videos, while remaining flexible for future upgrades.
- **What it does**: Outputs temporally consistent object tracks, each with a unique ID, trajectory, and motion statistics. These tracks are used for mask generation and final segmentation, and can be visualized for analysis.
- **Challenges**: Balancing sensitivity and inertia is key - too much inertia delays reactions to real changes, while too little causes instability and identity switches. Occlusions, rapid motion, and similar-looking objects can still challenge the tracker. The pipeline uses configurable parameters and heuristics to optimize tracking for each video.
- **Impact**: Enhanced tracking is essential for maintaining object identities, handling occlusions, and producing smooth, reliable segmentation masks. It enables the pipeline to track multiple objects robustly across long video sequences.

---

### Step 6: Masking (Instance Segmentation)

- **Library**: `ultralytics` YOLO (segmentation models), OpenCV, NumPy
- **How it works**: For each frame with tracked objects, the pipeline runs a YOLO segmentation model to generate instance masks. Masks are filtered and refined by checking if the center of each tracked bounding box (with a threshold relative to its size) falls within the mask. Masks are upscaled, morphologically cleaned, and edge-guided snapping is applied to improve mask boundaries. All valid masks are combined to produce a per-frame foreground mask.
- **Why this method**: Instance segmentation is essential for separating moving objects from the background at the pixel level. Using YOLO segmentation models allows fast, unified detection and mask prediction. The mask filtering logic ensures that only masks corresponding to tracked objects are used, reducing false positives and improving mask quality.
- **What it does**: Outputs a dictionary of per-frame masks for tracked objects, which are used in the next step for background reconstruction and final foreground extraction.
- **Challenges**: Mask quality depends on the YOLO model and input video. Small, overlapping, or fast-moving objects may have coarse or inaccurate masks. The pipeline uses edge-guided snapping and adaptive thresholding to improve results, but further refinement may be needed for challenging scenes.
- **Impact**: Accurate instance masks are critical for high-quality segmentation and background reconstruction. This step enables pixel-level separation of moving objects, supporting downstream analysis and visualization.

---

### Step 7: Video Segmentation (Foreground/Background Separation)

- **Library**: OpenCV, NumPy
- **How it works**: For each frame, the pipeline uses the instance masks generated in Step 6 to separate the video into foreground and background. The background is reconstructed using OpenCV's inpainting, which fills masked regions with plausible background pixels. The foreground is extracted by masking out everything except the detected moving objects, resulting in a video where only the segmented objects are visible against a black background.
- **Why this method**: Inpainting provides a fast and effective way to reconstruct backgrounds, especially when objects are small or move slowly. Foreground extraction enables pixel-accurate analysis and visualization of moving objects, supporting downstream tasks like tracking, editing, or further segmentation.
- **What it does**: Outputs two videos per input: one with the reconstructed background (objects removed), and one with the extracted foreground (objects only). Also saves a summary of processing statistics and output file paths.
- **Challenges**: Inpainting can produce artifacts, especially in regions with high texture, motion blur, or large occlusions. Foreground extraction quality depends on mask accuracy. The pipeline uses binary masks and adaptive inpainting methods to improve results, but further refinement may be needed for challenging scenes.
- **Impact**: This step enables clear separation of moving objects from the background, supporting analysis, visualization, and downstream applications such as video editing, object tracking, or generative modeling.

---

### * Visualization Options

- **Built-in Visualizations**: The pipeline can generate step-by-step visualizations for each major stage, including object detection, background point selection, motion analysis, tracking, masking, and segmentation. These visualizations help users understand and debug the pipeline, and provide clear insight into how each step transforms the video data.
- **How it works**: For each step, annotated videos are created showing detections, tracked points, object tracks, masks, and segmentation results. These are saved in the `visualizations` folder within each output directory. Visualization can be enabled or disabled via the configuration or command-line flags.
- **Compare Option**: As part of the visualization suite, the pipeline supports a "compare" mode, which generates a side-by-side video showing the original input and the segmentation output (foreground/background separation) in sync. This is useful for qualitative evaluation and presentation.
- **Why Visualization Matters**: Visualizations are essential for interpreting results, diagnosing errors, and communicating findings. They make the pipeline accessible to both researchers and practitioners, and support rapid iteration and improvement.
- **Customization**: Visualization options can be customized or extended by modifying the `visualizer.py` module. Users can add new overlays, statistics, or comparison modes as needed for their application.

---

## Configuration

#### Step 1: Object Detection

- **instance_model_path**: Path to YOLO model weights. Use a larger model for higher accuracy, or a smaller one for speed. Custom weights adapt to your domain.
- **yolo_conf_threshold**: Minimum confidence for detections. Increase to reduce false positives (only high-confidence objects detected), decrease to detect more objects (may include noise).
- **max_history**: Number of frames to keep detection history. Higher values smooth out missed detections but may delay response to new objects.

#### Step 2: Background Point Detection

- **max_points**: Max background points to track. More points improve motion estimation but increase computation.
- **quality_level**: Minimum quality for feature points. Lower finds more (may include weak/noisy points), higher finds fewer but stronger points.
- **min_distance**: Minimum distance between points. Larger spreads points out, smaller packs them closer (may overlap).
- **blockSize**: Size of block for corner detection. Larger smooths detection, smaller is more sensitive.
- **useHarrisDetector**: Use Harris detector (true) or default (false). Harris is robust but slower.
- **k**: Harris detector parameter. Adjust for sensitivity if using Harris.

#### Step 3: Motion Analysis

- **ransac_reproj_threshold**: RANSAC threshold for homography. Lower is stricter (fewer inliers, less noise), higher tolerates more noise (may include outliers).
- **motion_threshold**: Minimum residual motion to consider a point as moving. Lower detects subtle motion (may include background), higher ignores small movements.
- **max_corners**: Max corners for optical flow. More points for accuracy, fewer for speed.
- **lk_winSize**: Window size for optical flow. Larger is robust to noise, smaller is faster.
- **lk_maxLevel**: Pyramid levels for optical flow. More levels handle larger motions.
- **lk_criteria_eps**: Termination epsilon for optical flow. Lower for precision, higher for speed.
- **lk_criteria_count**: Max iterations for optical flow. More for accuracy, fewer for speed.

#### Step 4: Adaptive Object Tracking

- **min_points_per_detection**: Minimum motion points inside a detection to consider it dynamic. Lower accepts more objects (may include static ones), higher is stricter (only strong movers).
- **iou_threshold**: IoU threshold for matching detections to tracks. Lower allows looser matches (may cause identity switches), higher is stricter (may miss matches).
- **max_objects**: Max objects to track. More for crowded scenes, fewer for speed.
- **min_motion_duration**: Minimum duration for a track to be valid. Lower accepts short-lived objects, higher requires persistence.
- **min_avg_points_per_frame**: Minimum average motion points per frame for a track. Lower for more tracks, higher for stricter filtering.
- **track_history_length**: Number of frames to keep track history. Higher for stable tracks, lower for responsiveness.

#### Step 5: StrongSORT-Inspired Enhanced Tracking

- **max_disappeared**: Max frames a track can be missing before removal. Higher keeps tracks longer (good for occlusions), lower removes quickly (good for fast scenes).
- **min_hits**: Minimum detections before a track is confirmed. Lower confirms quickly (may include false tracks), higher is more reliable (may miss short-lived objects).
- **strong_iou_threshold**: IoU threshold for StrongSORT matching. Lower for looser assignment, higher for stricter.
- **max_tracks**: Max number of tracks. More for crowded scenes, fewer for speed.
- **feature_threshold**: Appearance similarity threshold for matching. Lower for looser matching (may confuse similar objects), higher for stricter (may miss matches).

#### Step 6: Masking (Instance Segmentation)

- **instance_seg_model_path**: Path to YOLO segmentation model weights. Use different models for accuracy/speed tradeoff or custom domains.
- **yolo_seg_conf_threshold**: Minimum confidence for segmentation masks. Higher for cleaner masks, lower to include more objects (may include noise).
- **yolo_iou_threshold**: IoU threshold for mask assignment. Adjust for mask quality and overlap.

#### Step 7: Video Segmentation (Foreground/Background Separation)

- **use_telea_inpaint**: If true, use TELEA inpainting for background reconstruction (fast, good for small objects). If false, use NS (Navier-Stokes) method (better for large occlusions, slower).
- **background_mask_style**: Style of the background in segmentation output. Options:
  - `black`: Masked background is set to black (default).
  - `blur`: Masked background is blurred.
 
<img src="gifs/cat-girl_compare.gif" loop=infinite>

---

**How to Tune:**
Adjust these parameters in `config.json` to optimize for your video, scene complexity, and desired output. Lower thresholds generally increase sensitivity (more objects, more noise); higher thresholds increase specificity (fewer objects, cleaner results). Model paths let you swap in custom-trained weights for domain adaptation. Visualization options help debug and validate each step.

---

## Dependencies & Libraries

**Core Dependencies:**

- `opencv-python`: Computer vision (feature tracking, optical flow, inpainting, image ops)
- `numpy`: Array and matrix operations
- `torch`: Deep learning backend for YOLO and segmentation
- `ultralytics`: YOLO detection and segmentation (v8 and custom variants)
- `scikit-learn`: Clustering (DBSCAN) for motion analysis
- `filterpy`: Kalman filtering for object tracking
- `scipy`: Linear assignment (Hungarian matching) and scientific utilities
- `tqdm`: Progress bars for user feedback

**Optional/Advanced:**

- `matplotlib`: Visualization and debugging (optional, for custom plots)
- `pandas`: Data analysis (optional, for advanced logging or benchmarking)

**Notes:**

- All dependencies are listed in `requirements.txt` for easy installation.
- GPU support is recommended for `torch` and `ultralytics` for best performance.

---

## Limitations & Future Work

**Current Limitations:**

- The pipeline may slow down or lose accuracy in extremely crowded scenes, with many moving objects, or when large objects dominate the frame.
- Mask and background quality can be affected by rapid motion, occlusions, or limited visible background.
- Real-time and multi-camera support are not yet implemented.

### Future Directions

- Integrate advanced segmentation models and deep inpainting for better mask and background quality.
- Develop smarter background modeling and tracking for crowded or dynamic scenes.
- Optimize for real-time and multi-camera use cases.
