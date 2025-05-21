import os
import re
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

def remove_mov_files(scene_path: Path) -> List[str]:
    """Remove any .mov files in the scene directory and return list of removed files."""
    removed_files = []
    mov_files = list(scene_path.glob('*.mov'))
    
    for mov_file in mov_files:
        try:
            mov_file.unlink()
            removed_files.append(mov_file.name)
        except Exception as e:
            print(f"Error removing {mov_file}: {e}")
    
    return removed_files

def validate_video_files(scene_path: Path) -> Tuple[bool, List[str]]:
    """Validate video files in a scene directory."""
    issues = []
    video_files = list(scene_path.glob('*.mp4'))
    mov_files = list(scene_path.glob('*.mov'))
    
    if mov_files:
        removed = remove_mov_files(scene_path)
        issues.append(f"Removed .mov files: {', '.join(removed)}")
    
    if len(video_files) == 1:
        if video_files[0].name != 'video.mp4':
            issues.append(f"Single video file should be named 'video.mp4', found: {video_files[0].name}")
    elif len(video_files) == 5:
        expected_files = {f"{i}.mp4" for i in range(1, 6)}
        found_files = {f.name for f in video_files}
        if found_files != expected_files:
            issues.append(f"Expected files 1.mp4 to 5.mp4, found: {', '.join(found_files)}")
    else:
        issues.append(f"Expected either 1 video.mp4 or 5 numbered videos (1.mp4 to 5.mp4), found {len(video_files)} files")
    
    return len(issues) == 0, issues

def process_video_directory(base_path: str) -> Dict[str, List[str]]:
    """Process all video directories and generate a report."""
    base_path = Path(base_path)
    report = {}
    
    if not base_path.exists():
        return {"Error": [f"Base directory {base_path} does not exist"]}
    
    for character_dir in base_path.iterdir():
        if not character_dir.is_dir():
            continue
            
        character_issues = []
        
        # Validate scene folders
        scenes_valid, scene_issues = validate_scene_folders(character_dir)
        character_issues.extend(scene_issues)
        
        # Validate video files in each scene
        for scene_dir in character_dir.iterdir():
            if scene_dir.is_dir() and re.match(r'scene\d+', scene_dir.name):
                videos_valid, video_issues = validate_video_files(scene_dir)
                if video_issues:
                    character_issues.append(f"In {scene_dir.name}: {'; '.join(video_issues)}")
        
        if character_issues:
            report[character_dir.name] = character_issues
        else:
            report[character_dir.name] = ["All valid"]
    
    return report

def print_report(report: Dict[str, List[str]]):
    """Print the validation report in a readable format."""
    print("\n=== Video Directory Structure Validation Report ===\n")
    
    for character, issues in report.items():
        print(f"\nCharacter: {character}")
        print("-" * (len(character) + 11))
        for issue in issues:
            print(f"- {issue}")
        print()

if __name__ == "__main__":
    videos_path = "videos_original"
    report = process_video_directory(videos_path)
    print_report(report)
