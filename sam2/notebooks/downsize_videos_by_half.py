import os
from moviepy.editor import VideoFileClip
from pathlib import Path

def resize_video(input_path, output_path):
    """
    Resize a video to half its original resolution
    """
    try:
        # Load the video
        video = VideoFileClip(input_path)
        
        # Get current resolution
        original_width, original_height = video.size
        print(f"Original resolution: {original_width}x{original_height}")
        
        # Calculate new resolution (half of original)
        new_width = original_width // 2
        new_height = original_height // 2
        print(f"New resolution: {new_width}x{new_height}")
        
        # Resize the video
        resized_video = video.resize((new_width, new_height))
        
        # Write the resized video
        resized_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Clean up
        video.close()
        resized_video.close()
        
        print(f"✅ Successfully resized: {os.path.basename(input_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_videos():
    """
    Process all videos in a user-specified folder and resize them to half resolution
    """
    # Ask user for folder path
    print("=== Video Resolution Reducer ===")
    print("This script will resize all videos in a folder to half their original resolution.")
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
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
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