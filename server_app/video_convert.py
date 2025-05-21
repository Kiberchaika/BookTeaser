import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def validate_scene_folders(character_path: Path) -> Tuple[bool, List[str]]:
    """Validate scene folders in a character directory."""
    issues = []
    scene_folders = [f for f in character_path.iterdir() if f.is_dir()]
    
    # Check if there are any non-scene folders
    invalid_folders = [f.name for f in scene_folders if not re.match(r'scene\d+', f.name)]
    if invalid_folders:
        issues.append(f"Found invalid folder names: {', '.join(invalid_folders)}")
    
    # Get scene numbers and check for gaps
    scene_numbers = []
    for folder in scene_folders:
        match = re.match(r'scene(\d+)', folder.name)
        if match:
            scene_numbers.append(int(match.group(1)))
    
    scene_numbers.sort()
    if scene_numbers:
        expected_numbers = list(range(1, max(scene_numbers) + 1))
        if scene_numbers != expected_numbers:
            issues.append(f"Scene numbers are not sequential. Found: {scene_numbers}")
    
    return len(issues) == 0, issues

def validate_video_files(scene_path: Path) -> Tuple[bool, List[str], List[Path]]:
    """Validate video files in a scene directory and return valid files."""
    issues = []
    valid_files = []
    video_files = list(scene_path.glob('*.mp4'))
    
    if len(video_files) == 1:
        if video_files[0].name == 'video.mp4':
            valid_files.append(video_files[0])
        else:
            issues.append(f"Single video file should be named 'video.mp4', found: {video_files[0].name}")
    elif len(video_files) == 5:
        expected_files = {f"{i}.mp4" for i in range(1, 6)}
        found_files = {f.name for f in video_files}
        if found_files == expected_files:
            valid_files.extend(video_files)
        else:
            issues.append(f"Expected files 1.mp4 to 5.mp4, found: {', '.join(found_files)}")
    else:
        issues.append(f"Expected either 1 video.mp4 or 5 numbered videos (1.mp4 to 5.mp4), found {len(video_files)} files")
    
    return len(issues) == 0, issues, valid_files

def convert_video(input_path: Path, output_path: Path) -> bool:
    """Convert video using ffmpeg with specified parameters."""
    try:
        print(f"Processing: {input_path}")
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(input_path),
            '-an',
            '-s', '1920x1080',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-y',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Completed: {input_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr.decode()}")
        return False

def process_video_directory(src_path: str, dst_path: str) -> Dict[str, List[str]]:
    """Process all video directories and convert valid files."""
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
        character_dst = dst_path / character_dir.name
        character_dst.mkdir(exist_ok=True)
        
        # Validate scene folders
        scenes_valid, scene_issues = validate_scene_folders(character_dir)
        character_issues.extend(scene_issues)
        
        # Process valid scenes
        for scene_dir in character_dir.iterdir():
            if scene_dir.is_dir() and re.match(r'scene\d+', scene_dir.name):
                videos_valid, video_issues, valid_files = validate_video_files(scene_dir)
                
                if videos_valid:
                    # Create scene directory in destination
                    scene_dst = character_dst / scene_dir.name
                    scene_dst.mkdir(exist_ok=True)
                    
                    # Convert valid files
                    for video_file in valid_files:
                        output_file = scene_dst / video_file.name
                        if not convert_video(video_file, output_file):
                            character_issues.append(f"Failed to convert {video_file.name} in {scene_dir.name}")
                else:
                    character_issues.append(f"In {scene_dir.name}: {'; '.join(video_issues)}")
        
        if character_issues:
            report[character_dir.name] = character_issues
        else:
            report[character_dir.name] = ["All files processed successfully"]
    
    return report

def print_report(report: Dict[str, List[str]]):
    """Print the processing report in a readable format."""
    print("\n=== Video Processing Report ===\n")
    
    for character, issues in report.items():
        print(f"\nCharacter: {character}")
        print("-" * (len(character) + 11))
        for issue in issues:
            print(f"- {issue}")
        print()

if __name__ == "__main__":
    src_path = "videos_original"
    dst_path = "videos_processed"
    report = process_video_directory(src_path, dst_path)
    print_report(report)