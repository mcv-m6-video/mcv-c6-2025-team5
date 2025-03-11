import cv2
import imageio
import os

def avi_to_gif(video_path, output_gif, start_frame=850, end_frame=920, fps=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frames = []
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_frame <= frame_id <= end_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_AREA)  # Downsample by 4
            frames.append(frame)
        
        frame_id += 1
        if frame_id > end_frame:
            break
    
    cap.release()
    
    if frames:
        imageio.mimsave(output_gif, frames, fps=fps)
        print(f"GIF saved as {output_gif}")
    else:
        print("Error: No frames captured.")

# Example usage
avi_to_gif("Week3/RAFT/vdo.avi", "output.gif")
