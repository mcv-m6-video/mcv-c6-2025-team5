import cv2
import os
import argparse

def images_to_video(image_folder, output_video, fps=30):
    # Get all image filenames sorted in order
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the folder!")
        return

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Add frame to the video

    video.release()
    print(f"Video saved as {output_video}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--videoout", required=False, default='./output_video.mp4', help="Path to the input video file.")
    parser.add_argument("--input", required=False, default='/home/marco/Documents/Week3_C6/frames_video/', help="Path to the output directory to save frames.")
    args = parser.parse_args()
    images_to_video(args.input, args.videoout , fps=30)
