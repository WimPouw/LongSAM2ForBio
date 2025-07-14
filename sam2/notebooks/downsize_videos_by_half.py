import os
import subprocess
import json
from pathlib import Path

def get_video_info(video_path):
    """
    Get video resolution using ffprobe
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                width = int(stream['width'])
                height = int(stream['height'])
                return width, height
        
        return None, None
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None, None

def resize_video(input_path, output_path):
    """
    Resize a video to half its original resolution using ffmpeg
    """
    try:
        # Get original resolution
        original_width, original_height = get_video_info(input_path)
        
        if original_width is None or original_height is None:
            print(f"❌ Could not get video info for: {os.path.basename(input_path)}")
            return False
        
        print(f"Original resolution: {original_width}x{original_height}")
        
        # Calculate new resolution (half of original)
        new_width = original_width // 2
        new_height = original_height // 2
        
        # Make sure dimensions are even (required by some codecs)
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1
        
        print(f"New resolution: {new_width}x{new_height}")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-crf', '23',  # Good quality-to-size ratio
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        # Run ffmpeg with minimal output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        print(f"✅ Successfully resized: {os.path.basename(input_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error processing {os.path.basename(input_path)}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def check_ffmpeg():
    """
    Check if ffmpeg is installed and available
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def process_videos():
    """
    Process all videos in a user-specified folder and resize them to half resolution
    """
    print("=== Video Resolution Reducer (FFmpeg) ===")
    print("This script will resize all videos in a folder to half their original resolution.")
    print()
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("❌ FFmpeg is not installed or not found in PATH.")
        print("Please install FFmpeg from https://ffmpeg.org/")
        print("Make sure both 'ffmpeg' and 'ffprobe' are in your system PATH.")
        input("\nPress Enter to exit...")
        return
    
    print("✅ FFmpeg found and ready to use!")
    print()
    
    while True:
        folder_path = input("Enter the path to the folder containing videos: ").strip()
        
        # Remove quotes if user wrapped the path in quotes
        if folder_path.startswith('"') and folder_path.endswith('"'):
            folder_path = folder_path[1:-1]
        if folder_path.startswith("'") and folder_path.endswith("'"):
            folder_path = folder_path[1:-1]
        
        input_path = Path(folder_path)
        
        # Check if folder exists
        if input_path.exists() and input_path.is_dir():
            break
        else:
            print(f"❌ Folder does not exist or is not a directory: {folder_path}")
            print("Please try again.")
            print()
    
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mts', '.m2ts'}
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {folder_path}")
        input("Press Enter to exit...")
        return
    
    print(f"\n📁 Found {len(video_files)} video files in: {input_path}")
    print("-" * 60)
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        # Create output filename with "_half" suffix
        output_file = video_file.parent / f"{video_file.stem}_half{video_file.suffix}"
        
        # Skip if output file already exists
        if output_file.exists():
            print(f"⏭️  Skipping (already exists): {output_file.name}")
            continue
        
        # Process the video
        if resize_video(str(video_file), str(output_file)):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful} videos")
    print(f"❌ Failed: {failed} videos")
    print(f"📁 Resized videos saved in: {input_path}")
    print("\n💡 Resized videos have '_half' added to their filename")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    process_videos()