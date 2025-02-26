import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path to the output directory to save frames.")
    args = parser.parse_args()

    train_dir = os.path.join(args.output, "frames_vdo_train25", "frames")
    eval_dir = os.path.join(args.output, "frames_vdo_eval75", "frames")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count <= 500:
            output_dir = train_dir
        else:
            output_dir = eval_dir

        frame_filename = os.path.join(output_dir, f"frame{frame_count:05d}.jpg")

        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames:")
    print(f" - Frames 0-500 saved to: {train_dir}")
    print(f" - Frames 501+ saved to: {eval_dir}")

if __name__ == "__main__":
    main()