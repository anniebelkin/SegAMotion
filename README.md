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

For more, you can download some saved results from [HERE](https://drive.google.com/file/d/154ihko5SqyqB4uqCkP6hRf9reFYgIMbB/view?usp=sharing)

<table>
  <tr>
    <td><img src="gifs/car-roundabout_full_compare.gif" loop=infinite></td>
  <td><img src="gifs/static_background_full_compare.gif" loop=infinite></td>
  </tr>
  <tr>
  <td><img src="gifs/horsejump-high_full_compare.gif" loop=infinite></td>
  <td><img src="gifs/bike-packing_full_compare.gif" loop=infinite></td>
  </tr>
  <tr>
    <td><img src="gifs/judo_full_compare.gif" loop=infinite></td>
    <td><img src="gifs/boxing-fisheye_full_compare.gif" loop=infinite></td>
  </tr>
</table>

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

See [PIPELINE.md](PIPELINE.md) for a detailed explanation of each pipeline step, including object detection, background point detection, motion analysis, tracking, masking, segmentation, and visualization options.

---

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for a full list of configuration options and tuning advice for each pipeline stage.

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
