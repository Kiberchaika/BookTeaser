import os
import subprocess
from pathlib import Path
from typing import List, Tuple

def get_video_files(scene_path: Path) -> List[Path]:
    """Get video files from a scene directory."""
    video_files = []
    
    # Check for video.mp4
    if (scene_path / 'video.mp4').exists():
        video_files.append(scene_path / 'video.mp4')
    # Check for numbered videos (1.mp4 to 5.mp4)
    else:
        video_files.append(scene_path / f"{1}.mp4")
    
    return video_files

def create_concat_file(video_files: List[Path], concat_file: Path) -> bool:
    """Create a file listing videos for ffmpeg concatenation."""
    try:
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video in video_files:
                f.write(f"file '{video.absolute()}'\n")
        return True
    except Exception as e:
        print(f"Error creating concat file: {e}")
        return False

def join_videos(video_files: List[Path], output_path: Path) -> bool:
    """Join multiple videos into a single file using ffmpeg."""
    if not video_files:
        print("No video files to join")
        return False
    
    # Create temporary concat file
    concat_file = output_path.parent / "temp_concat.txt"
    if not create_concat_file(video_files, concat_file):
        return False
    
    try:
        print(f"Joining videos into: {output_path}")
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully joined videos into: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error joining videos: {e.stderr.decode()}")
        return False
    finally:
        # Clean up temporary concat file
        if concat_file.exists():
            concat_file.unlink()

def process_directory(src_path: str, dst_path: str) -> dict:
    """Process all directories and join videos from scenes."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    report = {}
    
    if not src_path.exists():
        return {"Error": [f"Source directory {src_path} does not exist"]}
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    for character_dir in src_path.iterdir():
        if not character_dir.is_dir():
            continue
            
        character_issues = []
        character_dst = dst_path / f"{character_dir.name}.mp4"
        
        # Process scenes
        scene_videos = []
        # Get all scene directories and sort them by scene number
        scene_dirs = [d for d in character_dir.iterdir() if d.is_dir() and d.name.startswith('scene')]
        scene_dirs.sort(key=lambda x: int(x.name[5:]))  # Sort by number after 'scene'
        
        for scene_dir in scene_dirs:
            video_files = get_video_files(scene_dir)
            if video_files:
                scene_videos.extend(video_files)
            else:
                character_issues.append(f"No valid videos found in {scene_dir.name}")
        
        if scene_videos:
            if join_videos(scene_videos, character_dst):
                character_issues.append("Videos joined successfully")
            else:
                character_issues.append("Failed to join videos")
        else:
            character_issues.append("No valid videos found in any scene")
        
        report[character_dir.name] = character_issues
    
    return report

def print_report(report: dict):
    """Print the processing report in a readable format."""
    print("\n=== Video Joining Report ===\n")
    
    for character, issues in report.items():
        print(f"\nCharacter: {character}")
        print("-" * (len(character) + 11))
        for issue in issues:
            print(f"- {issue}")
        print()

if __name__ == "__main__":
    src_path = "videos_processed"  # Use the processed videos directory
    dst_path = "videos_joined"
    report = process_directory(src_path, dst_path)
    print_report(report) 