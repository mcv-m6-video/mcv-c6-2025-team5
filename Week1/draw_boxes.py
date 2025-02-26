import cv2
from bs4 import BeautifulSoup
import numpy as np
import argparse

def read_xml(path_xml):
    # Read the XAML file
    with open(path_xml, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Parse the XAML content
    soup = BeautifulSoup(xml_content, "xml")

    # Dictionary to store boxes per frame
    frames_data = {}

    # Find all 'box' elements
    boxes = soup.find_all("box")

    for box in boxes:
        frame = int(box.get("frame"))

        # Get bounding box coordinates
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        if frame not in frames_data:
            frames_data[frame] = []
        
        # Add box information for this frame
        frames_data[frame].append((xtl, ytl, xbr, ybr))
    
    return frames_data

def color_selector(n):
    colors = [
        (255, 0, 0),     # Red
        (0, 0, 255),     # Blue
        (255, 165, 0),   # Orange
        (128, 0, 128),   # Purple
        (255, 69, 0),    # Red-Orange
        (75, 0, 130),    # Indigo
        (255, 192, 203), # Pink
        (139, 0, 0),     # Dark Red
        (255, 140, 0),   # Dark Orange
        (0, 0, 139)      # Dark Blue
    ]
    return colors[n % 10]

def video_creation(gt_bb, results_bb, in_video_path, out_video_path='output_video_with_boxes.mp4', display=False):
    # Open the video
    cap = cv2.VideoCapture(in_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter to save the output video (optional)
    output_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # Draw boxes for the current frame
        if frame_number in gt_bb:
            for box in gt_bb[frame_number]:
                xtl, ytl, xbr, ybr = box

                # Draw rectangle (bounding box) on the frame
                cv2.rectangle(frame, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (0, 255, 0), 2)  # Green box with thickness 2

        for i,(_, result_bb) in enumerate(results_bb):
            if frame_number in result_bb:
                for box in result_bb[frame_number]:
                    xtl, ytl, xbr, ybr = box

                    # Draw rectangle (bounding box) on the frame
                    cv2.rectangle(frame, (int(xtl), int(ytl)), (int(xbr), int(ybr)), color_selector(i), 2)  # Color box with thickness 2

        # Write the frame with boxes to the output video
        output_video.write(frame)
        
        # Optional: Display the frame
        if display:
            cv2.imshow('Frame with Bounding Boxes', frame)

            # Break the loop if you press 'q' (quit) during the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_number += 1

    # Release the video capture and writer objects
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("Video with bounding boxes saved successfully!")

def legend_creation(results_bb, image_size=(250, 200), save_path="output_legend.png"):

    box_size = 20  # Size of each color box
    padding = 10   # Space around the legend
    spacing = 40   # Space between each label row

    # Calculate image height dynamically based on the number of elements
    img_width = 300  # Adjust width as needed
    img_height = (len(results_bb)+1) * spacing + padding * 2

    # Create a white background
    legend = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  

    x, y = padding, padding  # Start position

    for i, (method, _) in enumerate(results_bb):
        label = f"Method: {method}"
        color = color_selector(i)
        # Draw color box
        cv2.rectangle(legend, (x, y), (x + box_size, y + box_size), color, -1)  

        # Draw text label next to the color box
        cv2.putText(legend, label, 
                    (x + box_size + 10, y + box_size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text for contrast
        
        y += spacing  # Move down for the next label

    label = "Ground truth"
    color = (0,255,0)
    # Draw color box
    cv2.rectangle(legend, (x, y), (x + box_size, y + box_size), color, -1)  
    # Draw text label next to the color box
    cv2.putText(legend, label, 
                (x + box_size + 10, y + box_size - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text for contrast
    
    y += spacing  

    # Save the image
    cv2.imwrite(save_path, legend)
    print(f"Legend saved as {save_path}")

def parse_pairs(pairs_str):
    pairs = []
    for pair in pairs_str:
        # Remove parentheses and split by comma
        description, path = pair.strip("()").split(",", 1)  # Split only on the first comma
        pairs.append((description.strip(), path.strip()))  # Strip extra spaces
    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./week1_anot.xml", help='Input xml')
    parser.add_argument('--path_video', required=False, default="./AICity_data/train/S03/c010/vdo.avi", help='Input xml')
    parser.add_argument("--pairs", required=True, type=str, nargs="+", help="List of (method_name, xml_path) pairs")
    args = parser.parse_args()

    gt_bb = read_xml(args.path_xml)
    xml_paths = parse_pairs(args.pairs)
    results_bb = []

    for method_name, path in xml_paths:
        tmp = read_xml(path)
        results_bb.append((method_name, tmp))
    
    video_creation(gt_bb, results_bb, args.path_video)
    legend_creation(results_bb)

    # TODO: Compute MAP for each method.


    