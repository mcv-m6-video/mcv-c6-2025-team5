import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights).to(DEVICE).eval()

def load_frame(frame):
    image = Image.fromarray(frame)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((520, 960))  # Resize to match RAFT expected input size
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def flow_to_color(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_video(video_path, output_flow_dir, output_video_path):
    os.makedirs(output_flow_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return
    
    frame_idx = 0
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        image1 = load_frame(prev_frame)
        image2 = load_frame(next_frame)
        
        with torch.no_grad():
            flow = model(image1, image2)[0]
            flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        np.save(os.path.join(output_flow_dir, f'flow_{frame_idx:04d}.npy'), flow)
        
        hsv_flow = flow_to_color(flow)
        
        if out is None:
            h, w, _ = hsv_flow.shape
            out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        
        out.write(hsv_flow)
        
        prev_frame = next_frame
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Optical flow video saved to {output_video_path}")

def main():
    video_path = '/ghome/c5mcv05/c03-10/vdo.avi'
    output_flow_dir = 'output_flow'
    output_video_path = 'optical_flow_visual.avi'
    
    process_video(video_path, output_flow_dir, output_video_path)

if __name__ == "__main__":
    main()
