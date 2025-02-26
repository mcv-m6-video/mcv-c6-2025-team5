import cv2
import numpy as np
import argparse
import sys
import os
from lxml import etree 
import scipy



def  video_loader_splitter(path_video, video_init=0.25, color=True):
    cap = cv2.VideoCapture(path_video)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    # Process each frame
    frame_number = 0
    frame_struct = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        if color:
            result_frame = frame.copy() #shape (1080,1920,3)
        else:
            result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#shape (1080,1920)
        
        if frame_number == 0:
            print(f"Shape of the frame:{result_frame.shape}")
        
        frame_struct.append(result_frame)
        frame_number += 1


    # Release the video capture and writer objects
    cap.release()
    cv2.destroyAllWindows()

    perct = round(video_init*len(frame_struct))
    print(f"Frame struct: {len(frame_struct)}")
    return frame_struct[:perct], frame_struct[perct:],perct


def compute_background_model(init_frames, output_dir):
    init_cube = np.array(init_frames)
    print(f"Frame cube shape: {init_cube.shape}")
    mean_cube = np.mean(init_cube, axis=0)
    std_cube = np.std(init_cube, axis=0)

    print(f"Mean cube shape: {mean_cube.shape}")
    print(f"Std cube shape: {std_cube.shape}")

    return mean_cube, std_cube

def compute_adaptive_background(frames, init_mean, init_std, alpha, ro, save_path):
    os.makedirs(save_path, exist_ok=True)  

    mean = init_mean.copy()
    std = init_std.copy()
    fg_masks = []


    for i, frame in enumerate(frames):
        diff = abs(frame - mean)  
        threshold = alpha * (std + 2)

        fg_mask = np.where(diff >= threshold, 1, 0)  

        if len(fg_mask.shape) == 3:  
            fg_mask = np.all(fg_mask, axis=2).astype(int)

        mean = (1 - ro) * mean + ro * frame
        std = np.sqrt((1 - ro) * std**2 + ro * (frame - mean) ** 2)

        fg_masks.append(fg_mask)

    return fg_masks



def process_mask(mask, threshold=9, min_area=400):
    result = []
    for frame in mask:
        bb_frame_result = []

        binary_image = (frame * 255).astype(np.uint8)

        smoothed = scipy.ndimage.gaussian_filter(binary_image.astype(float), sigma=12)
        smoothed = scipy.ndimage.grey_dilation(smoothed, size=(5, 5))
        mask_filtered = (smoothed >= threshold).astype(np.uint8)
        mask_filtered = scipy.ndimage.binary_fill_holes(mask_filtered).astype(np.uint8)

        # Get connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filtered, connectivity=8)

        for i in range(1, num_labels):  # Start from 1 to skip the background label (0)
            x, y, w, h, area = stats[i]
            if area >= min_area:  
                bb = (x, y, x + w, y + h)
                bb_frame_result.append(bb)

        result.append(bb_frame_result)
    
    return result



def write_xml(struct, output_path, init_i_frame):
     # Now create a new XML structure with frames as the first level
    root = etree.Element("annotations")

    for i_frame, frame in enumerate(struct):
        real_i_frame = i_frame + init_i_frame
        frame_element = etree.SubElement(root, "frame", number=str(real_i_frame))

        for (xtl, ytl, xbr, ybr) in frame:
            box_element = etree.SubElement(frame_element, "box", frame=str(real_i_frame), 
                                          xtl=str(xtl), ytl=str(ytl),
                                          xbr=str(xbr), ybr=str(ybr))

    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./output_boxes.xml", help='Output XML path')
    parser.add_argument('--path_video', required=False, default="./AICity_data/train/S03/c010/vdo.avi", help='Input video path')
    parser.add_argument('--alpha', required=False, type=float, default=9.0, help='Alpha factor')
    parser.add_argument('--ro', required=False, type=float, default=0.6, help='Ro factor')
    parser.add_argument('--video_init', required=False, type=float, default=0.25, help='Initial percentage of video to use')
    parser.add_argument('--output_dir', required=False, default="./output_masks_refine", help='Directory for output masks')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_frames, result_frames, split_frame_number = video_loader_splitter(args.path_video, args.video_init)
    mean, std = compute_background_model(init_frames, args.output_dir)
    print("computed background model")
    unrefined_mask = compute_adaptive_background(result_frames, mean, std, args.alpha, args.ro, args.output_dir)
    result = process_mask(unrefined_mask)
    write_xml(result, args.path_xml, split_frame_number)
    print("generated bboxes")

