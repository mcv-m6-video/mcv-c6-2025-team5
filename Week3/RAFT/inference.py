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


IMAGE1_PATH = '000045_10.png'
IMAGE2_PATH = '000045_11.png'
OUTPUT_PATH = 'optical_flow.png'
OUTPUT_NPY_PATH = 'optical_flow.npy'

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
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


def run_raft(image1_path, image2_path, output_path='optical_flow.png', output_npy_path='optical_flow.npy'):
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    with torch.no_grad():
        flow = model(image1, image2)[0]  # Output is (flow_u, flow_v)
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 2)

        np.save(output_npy_path, flow)
        print(f"Optical flow saved as .npy to {output_npy_path}")

    h, w = image1.shape[2], image1.shape[3]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = 255
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    cv2.imwrite(output_path, rgb)
    print(f"Optical flow visual saved to {output_path}")

def main():
    run_raft(IMAGE1_PATH, IMAGE2_PATH, OUTPUT_PATH, OUTPUT_NPY_PATH)

if __name__ == "__main__":
    main()
