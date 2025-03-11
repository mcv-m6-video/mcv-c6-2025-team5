import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video", required=False, default='/mnt/home/C6/AICity_data/train/S03/c010/vdo.avi', help="Path to the input video file.")
    parser.add_argument("--output1", required=False, default='/mnt/datain/', help="Path to the output directory to save frames.")
    parser.add_argument("--output2", required=False, default='/mnt/dataout/', help="Path to the output directory to save frames.")
    args = parser.parse_args()

    frames_dir1 = os.path.join(args.output1)
    os.makedirs(frames_dir1, exist_ok=True)

    frames_dir2 = os.path.join(args.output2)
    os.makedirs(frames_dir2, exist_ok=True)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename1 = os.path.join(frames_dir1, f"frame{frame_count:06d}.png")
        if frame_count>=0 and frame_count<1000:
            cv2.imwrite(frame_filename1, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames:")
    print(f"Frames saved to: {frames_dir1}")

if __name__ == "__main__":
    main()