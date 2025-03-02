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
    """Export frames to video with fallback mechanisms."""
    try:
        # Try the recommended way first
        from diffusers.utils import export_to_video
        export_to_video(frames, output_file)
        print(f"Video exported to {output_file}")
    except Exception as e:
        print(f"Error using export_to_video: {e}")
        try:
            # Try alternative with imageio directly
            import imageio
            imageio.mimsave(output_file, frames)
            print(f"Video exported to {output_file} using imageio")
        except Exception as e:
            print(f"Error using imageio: {e}")
            try:
                # Fallback to OpenCV
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
            except Exception as e:
                print(f"All video export methods failed: {e}")
                print("Skipping video export - GIF should still be available")