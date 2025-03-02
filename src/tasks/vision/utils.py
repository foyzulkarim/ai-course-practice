import torch

def get_device():
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

def export_video_robust(frames, output_file):
    """Export frames to video with fallback mechanisms.
    
    Attempts multiple methods to export frames to a video file:
    1. Using diffusers.utils.export_to_video (primary method)
    2. Using imageio.mimsave as fallback
    3. Using OpenCV VideoWriter as final fallback
    
    Args:
        frames: List of image frames to export
        output_file: Path to save the video file
        
    Returns:
        bool: True if export succeeded with any method, False if all failed
    """
    # Try with diffusers export_to_video
    if _try_export_diffusers(frames, output_file):
        return True
        
    # Try with imageio
    if _try_export_imageio(frames, output_file):
        return True
        
    # Try with OpenCV
    if _try_export_opencv(frames, output_file):
        return True
        
    # All methods failed
    print(f"All video export methods failed")
    print("Skipping video export - GIF should still be available")
    return False
    
def _try_export_diffusers(frames, output_file):
    """Try exporting video using diffusers library."""
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_file)
        print(f"Video exported to {output_file}")
        return True
    except Exception as e:
        print(f"Error using export_to_video: {e}")
        return False
        
def _try_export_imageio(frames, output_file):
    """Try exporting video using imageio library."""
    try:
        import imageio
        imageio.mimsave(output_file, frames)
        print(f"Video exported to {output_file} using imageio")
        return True
    except Exception as e:
        print(f"Error using imageio: {e}")
        return False
        
def _try_export_opencv(frames, output_file):
    """Try exporting video using OpenCV library."""
    try:
        import cv2
        import numpy as np
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, 8, (width, height))
        
        for frame in frames:
            # Convert from RGB to BGR
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Video exported to {output_file} using OpenCV")
        return True
    except Exception as e:
        print(f"Error using OpenCV: {e}")
        return False