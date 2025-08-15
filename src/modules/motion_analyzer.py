import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    required_keys = [
        "ransac_reproj_threshold", "motion_threshold", "max_corners",
        "lk_winSize", "lk_maxLevel", "lk_criteria_eps", "lk_criteria_count"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config

class MotionAnalyzer:
    def __init__(self, config_path="config.json"):
        config = load_config(config_path)
        self.config = config
        self.ransac_thresh = config["ransac_reproj_threshold"]
        self.motion_thresh = config["motion_threshold"]
        self.max_corners = config["max_corners"]
        self.lk_params = dict(
            winSize=tuple(config["lk_winSize"]),
            maxLevel=config["lk_maxLevel"],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config["lk_criteria_count"], config["lk_criteria_eps"])
        )

    def load_background_points(self, json_path):
        data = json.loads(Path(json_path).read_text())
        bg = {}
        for fidx, frame_data in data['background_points'].items():
            pts = frame_data.get('points', [])
            bg[int(fidx)] = np.array(pts, dtype=np.float32).reshape(-1, 2)
        return bg

    def warp_points(self, pts, H):
        if len(pts) == 0:
            return pts
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        warped = (H @ pts_h.T).T
        warped /= warped[:, 2:3]
        return warped[:, :2]

    def process_video(self, video_path, background_json, output_dir, visualize=True):
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bg_pts = self.load_background_points(background_json)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = {
            "video_info": {"path": str(video_path), "frame_count":n_frames, "fps":fps},
            "frames": {}
        }

        # Read first frame
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for f in tqdm(range(n_frames-1), desc="Step 3: Analyzing motion"):
            # Read next frame
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Track background points
            bg_t = bg_pts.get(f,    np.empty((0,2),dtype=np.float32))
            bg_t1= bg_pts.get(f+1,  np.empty((0,2),dtype=np.float32))
            if len(bg_t)>=4 and len(bg_t1)>=4:
                # use LK to get accurate bg_t1 from prev_gray→gray
                bg_tracked, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, bg_t.reshape(-1,1,2), None, **self.lk_params)
                mask = st.reshape(-1)==1
                src = bg_t[mask]; dst = bg_tracked.reshape(-1,2)[mask]
                if len(src)>=4:
                    H_bg, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
                else:
                    H_bg = None
            else:
                H_bg = None

            # Detect and track all feature points
            pts_t = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=self.max_corners,
                qualityLevel=0.01, minDistance=7, blockSize=7
            )
            pts_t = pts_t.reshape(-1,2) if pts_t is not None else np.empty((0,2),np.float32)

            if len(pts_t)>0:
                pts_t1, st2, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts_t.reshape(-1,1,2), None, **self.lk_params)
                mask2 = st2.reshape(-1)==1
                pts_t  = pts_t[mask2]
                pts_t1 = pts_t1.reshape(-1,2)[mask2]
            else:
                pts_t1 = np.empty((0,2),np.float32)

            # 3) Predict where pts_t “should” go by bg homography
            if H_bg is not None:
                predicted = self.warp_points(pts_t, H_bg)
            else:
                predicted = pts_t.copy()

            # 4) Residual: actual vs. predicted
            if len(pts_t1)>0:
                dists = np.linalg.norm(pts_t1 - predicted, axis=1)
                moving = pts_t[dists>self.motion_thresh]
            else:
                moving = np.empty((0,2),np.float32)

            # 5) Save results for this frame (t)
            results["frames"][str(f)] = {
                "num_tracked": int(len(pts_t)),
                "num_moving": int(len(moving)),
                "moving_points": moving.tolist()
            }

            # shift for next iteration
            prev_gray = gray
            prev_frame= frame

        cap.release()

        # dump JSON
        (out_dir/"moving_points_results.json").write_text(
            json.dumps(results, indent=2)
        )

        return results
