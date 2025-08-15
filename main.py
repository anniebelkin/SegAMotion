import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
import cv2

from src.modules.object_detector import ObjectDetector
from src.modules.background_detector import BackgroundPointDetector
from src.modules.motion_analyzer import MotionAnalyzer
from src.modules.object_filter import ObjectFilter
from src.modules.strong_filter import StrongFilter
from src.modules.masking import VideoMasker
from src.modules.segment import VideoSegmenter
from src.utils.visualizer import visualizer

class PipelineController:
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_folder = self.project_root / "data" / "vid"
        self.output_folder = self.project_root / "output"
        self.compare_output_folder = self.project_root / "compare_output"
        
    def find_video_paths(self, video_names):
        video_paths = []
        
        for video_name in video_names:
            video_path = Path(video_name)
            
            if video_path.is_absolute() and video_path.exists():
                video_paths.append(video_path)
            elif (self.project_root / video_name).exists():
                video_paths.append(self.project_root / video_name)
            elif (self.data_folder / video_name).exists():
                video_paths.append(self.data_folder / video_name)
            else:
                print(f"Warning: Video not found: {video_name}")
        
        return video_paths
    
    def get_all_videos_in_data_folder(self):
        if not self.data_folder.exists():
            print(f"Data folder not found: {self.data_folder}")
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = set()
        
        for ext in video_extensions:
            video_files.update(self.data_folder.glob(f"*{ext}"))
            video_files.update(self.data_folder.glob(f"*{ext.upper()}"))
        
        return sorted(list(video_files))
    
    def create_step_visualizations(self, video_path, output_dir, enable_visualizations=True):
        """Create visualization videos for each pipeline step"""
        if not enable_visualizations:
            return
            
        print(f"Creating visualizations for {video_path.name}...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # This will be called after each step to generate visualization videos
        # Implementation will be added as we integrate with each step
        pass
    
    def run_pipeline(self, video_paths, **config):
        results = []
        
        for video_path in video_paths:
            print(f"Processing: {video_path.name}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_folder / video_path.stem / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_result = {'video_path': str(video_path), 'output_directory': str(output_dir)}
            
            try:
                # Set visualizer context once per video
                if config.get('create_visualizations', True):
                    visualizer.set_video_context(str(video_path), str(output_dir))
                
                # Step 1: Object Detection
                detector = ObjectDetector()
                detector.process_video(video_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(1)
                
                # Step 2: Background Points
                detection_results_path = output_dir / "detection_results.json"
                bg_detector = BackgroundPointDetector()
                bg_detector.process_video(video_path, detection_results_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(2)
                
                # Step 3: Motion Analysis
                background_results_path = output_dir / "background_points_results.json"
                motion_analyzer = MotionAnalyzer()
                motion_analyzer.process_video(video_path, background_results_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(3)
                
                # Step 4: Object Tracking
                motion_results_path = output_dir / "moving_points_results.json"
                obj_filter = ObjectFilter()
                obj_filter.process_video(video_path, motion_results_path, detection_results_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(4)
                
                # Step 5: Strong Tracking
                step4_results_path = output_dir / "object_tracking_results.json"
                strong_tracker = StrongFilter()
                strong_tracker.process_video(video_path, step4_results_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(5)
                
                # Step 6: Video Masking
                step5_results_path = output_dir / "strong_tracking_results.json"
                masker = VideoMasker()
                masking_data = masker.process_video(video_path, step5_results_path, output_dir)
                if config.get('create_visualizations', True):
                    visualizer.create_step_visualization(6, masking_data)
                
                # Step 7: Video Segmentation (Background/Foreground)
                segmenter = VideoSegmenter()
                segmenter.process_video(video_path, masking_data, output_dir)
                
                video_result['status'] = 'success'
                
                # Handle compare functionality
                if config.get('compare', True):
                    background_video = output_dir / f"{video_path.stem}_background.mp4"
                    foreground_video = output_dir / f"{video_path.stem}_foreground.mp4"
                    destination = self.compare_output_folder / f"{video_path.stem}_compare.mp4"
                    visualizer.merge_videos_side_by_side(
                        str(background_video), 
                        str(foreground_video), 
                        str(destination),
                        "Background", "Foreground"
                    )

            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                video_result['status'] = 'failed'
                video_result['error'] = str(e)
            
            results.append(video_result)
        
        # Ensure all visualizations are complete before ending
        visualizer.wait_for_completion()
        
        return results

    def clean_output_folder(self):
        if self.output_folder.exists():
            shutil.rmtree(self.output_folder)
            print("Output folder cleaned.")
        else:
            print("Output folder doesn't exist.")

    def clean_compare_output_folder(self):
        if self.compare_output_folder.exists():
            shutil.rmtree(self.compare_output_folder)
            print("Compare output folder cleaned.")
        else:
            print("Compare output folder doesn't exist.")

    def clean_all_folders(self):
        self.clean_output_folder()
        self.clean_compare_output_folder()


def main():
    parser = argparse.ArgumentParser(description='Video Analysis Pipeline')
    parser.add_argument('--path', nargs='*', help='Video paths (default: all videos in data/vid)')
    parser.add_argument('--compare', action='store_true', help='Create comparison videos')
    parser.add_argument('--clean', action='store_true', help='Clean output folder')
    parser.add_argument('--cleanall', action='store_true', help='Clean all folders')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization video creation')

    args = parser.parse_args()
    controller = PipelineController()

    if args.cleanall:
        controller.clean_all_folders()
        return 0
    
    if args.clean:
        controller.clean_output_folder()
        return 0

    # Load configuration
    config_path = Path('config.json')
    if not config_path.exists():
        print('Error: config.json not found!')
        return 1
        
    with open(config_path) as f:
        config = json.load(f)

    # Get video paths
    if args.path is None:
        video_paths = controller.get_all_videos_in_data_folder()
        if not video_paths:
            print("No videos found in data/vid folder")
            return 1
    else:
        video_paths = controller.find_video_paths(args.path)
        if not video_paths:
            print("No valid video paths found")
            return 1

    # Run the pipeline
    config['compare'] = args.compare
    config['create_visualizations'] = not args.no_viz
    results = controller.run_pipeline(video_paths, **config)
    
    print("Pipeline completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
