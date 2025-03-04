import cv2
import os

def video_to_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when no more frames

        frame_path = os.path.join(output_folder, f"{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1

    cap.release()
    print(f"Frames saved in: {output_folder}")

# Example usage
video_path = "./AICity_data/train/S03/c010/vdo.avi"  
output_folder = "./dataset/images"

video_to_frames(video_path, output_folder)
