import cv2
import numpy as np
import argparse
import sys
import os
from lxml import etree 

def video_loader_splitter(path_video, video_init=0.25, color=True):
    cap = cv2.VideoCapture(path_video)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = 0
    frame_struct = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if color:
            result_frame = frame.copy()
        else:
            result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_number == 0:
            print(f"Shape of the frame: {result_frame.shape}")
        
        frame_struct.append(result_frame)
        frame_number += 1

        if frame_number > 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    perct = round(video_init * len(frame_struct))
    print(f"Frame struct: {len(frame_struct)}")
    return frame_struct[:perct], frame_struct[perct:], perct, fps

def compute_background_model(init_frames, output_dir):
    init_cube = np.array(init_frames)
    print(f"Frame cube shape: {init_cube.shape}")
    mean_cube = np.mean(init_cube, axis=0)
    std_cube = np.std(init_cube, axis=0)

    print(f"Mean cube shape: {mean_cube.shape}")
    print(f"Std cube shape: {std_cube.shape}")

    cv2.imwrite(os.path.join(output_dir, "background_mean.png"), mean_cube.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "background_std.png"), (std_cube * 255 / np.max(std_cube)).astype(np.uint8))

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
        cv2.imwrite(f"{save_path}/frame_{i:04d}.png", fg_mask * 255)  

    return fg_masks

def refine_frames(unrefined_mask):
    result = []
    kernel = np.ones((5, 5), np.uint8)
    for frame_mask in unrefined_mask:
        tmp_f = frame_mask.astype(np.uint8)
        tmp_f = cv2.morphologyEx(tmp_f, cv2.MORPH_OPEN, kernel)
        tmp_f = cv2.morphologyEx(tmp_f, cv2.MORPH_CLOSE, kernel)
        result.append(tmp_f)
    return result

def write_xml(struct, output_path, init_i_frame):
    root = etree.Element("annotations")

    for i_frame, frame in enumerate(struct):
        real_i_frame = i_frame + init_i_frame
        frame_element = etree.SubElement(root, "frame", number=str(real_i_frame))

        for (xtl, ytl, xbr, ybr) in frame:
            etree.SubElement(frame_element, "box", frame=str(real_i_frame), 
                             xtl=str(xtl), ytl=str(ytl),
                             xbr=str(xbr), ybr=str(ybr))

    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def box_generator(mask, output_path, i_frame):
    result = []
    for frame in mask:
        bb_frame_result = []
        binary_image = 255 * frame
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bb = (x, y, x + w, y + h)
            bb_frame_result.append(bb)
        
        result.append(bb_frame_result)
    
    write_xml(result, output_path, i_frame)

def save_video_from_masks(mask_folder, output_video_path, fps):
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])
    if not mask_files:
        print("No mask frames found.")
        return
    
    sample_frame = cv2.imread(os.path.join(mask_folder, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = sample_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    for mask_file in mask_files:
        frame = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        out.write(frame)
    
    out.release()
    print(f"Saved video: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./output_boxes.xml", help='Output XML path')
    parser.add_argument('--path_video', required=False, default="./AICity_data/train/S03/c010/vdo.avi", help='Input video path')
    parser.add_argument('--alpha', required=False, type=float, default=2.0, help='Alpha factor')
    parser.add_argument('--ro', required=False, type=float, default=0.1, help='Ro factor')
    parser.add_argument('--video_init', required=False, type=float, default=0.25, help='Initial percentage of video to use')
    parser.add_argument('--output_dir', required=False, default="./output_masks", help='Directory for output masks')
    parser.add_argument('--output_video', required=False, default="./output_masks/masks.avi", help='Output video path')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_frames, result_frames, split_frame_number, fps = video_loader_splitter(args.path_video, args.video_init)
    mean, std = compute_background_model(init_frames, args.output_dir)
    unrefined_mask = compute_adaptive_background(result_frames, mean, std, args.alpha, args.ro, args.output_dir)
    
    mask = refine_frames(unrefined_mask)
    box_generator(mask, args.path_xml, split_frame_number)

    #save_video_from_masks(args.output_dir, args.output_video, fps)
