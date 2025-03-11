import numpy as np
import cv2
import os
import time
import pyflow
from PIL import Image
import imageio

def flow_to_color(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_video(video_path, output_flow_dir, output_gif_path, gif_start=25+60, gif_end=32+60):
    os.makedirs(output_flow_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int((gif_start - 1) * fps)  # Start from one frame before 25s
    end_frame = int(gif_end * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return
    
    print("Saving first frame...")
    cv2.imwrite("first_frame.png", prev_frame)
    
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0
    
    frames = []
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        im1 = prev_frame.astype(float) / 255.
        im2 = next_frame.astype(float) / 255.
        
        s = time.time()
        u, v, _ = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        e = time.time()
        print(f'Time Taken: {e - s:.2f} seconds for frame {frame_idx}')
        
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        np.save(os.path.join(output_flow_dir, f'flow_{frame_idx:04d}.npy'), flow)
        
        hsv_flow = flow_to_color(flow)
        frames.append(hsv_flow)
        
        prev_frame = next_frame
        frame_idx += 1
    
    cap.release()
    
    print("Saving last frame...")
    cv2.imwrite("last_frame.png", prev_frame)
    
    if frames:
        imageio.mimsave(output_gif_path, frames, format='GIF', fps=10)
        print(f"GIF saved to {output_gif_path}")

def main():
    video_path ='/ghome/c5mcv05/c03-10/vdo.avi'
    output_flow_dir = 'output_flow'
    output_gif_path = 'optical_flow.gif'
    
    process_video(video_path, output_flow_dir, output_gif_path)

if __name__ == "__main__":
    main()