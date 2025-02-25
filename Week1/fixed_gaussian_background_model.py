import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys
matplotlib.use('TkAgg')  # Ensure interactive backend

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

        if frame_number > 100:
            break

    # Release the video capture and writer objects
    cap.release()
    cv2.destroyAllWindows()

    perct = round(video_init*len(frame_struct))
    print(f"Frame struct: {len(frame_struct)}")
    return frame_struct[:perct], frame_struct[perct:],perct

def compute_background_model(init_frames):
    init_cube = np.array(init_frames)
    print(f"Frame cube shape: {init_cube.shape}")
    mean_cube = np.mean(init_cube, axis=0)
    std_cube = np.std(init_cube, axis=0)
    
    print(f"Mean cube shape: {mean_cube.shape}")
    print(f"std cube shape: {std_cube.shape}")

    return mean_cube,std_cube


def compute_background(frames, mean, std, alpha):
    result = []
    down_limit = alpha*(std+2)

    for frame in frames:

        exp = abs(frame-mean) #|I-mean| >= alpha(std+2)
        exp = exp - down_limit 
        exp[exp >= 0] = 1 
        exp[exp < 0] = 0

        if len(exp.shape) == 3:
            exp = np.all(exp, axis=2).astype(int)
            if not len(exp.shape) == 2:
                sys.exit(1)
        print(f"Shape of original pixel mask:{exp.shape}")
        plt.imshow(exp)
        plt.show()
        result.append(exp)
    
    return result

def refine_frames(unrefined_mask):
    result = []
    kernel = np.ones((5, 5), np.uint8)
    for frame_mask in unrefined_mask:
        tmp_f = frame_mask.astype(np.uint8)
        # we remove small white objects
        tmp_f = cv2.morphologyEx(tmp_f, cv2.MORPH_OPEN, kernel)
        # remove small black artifacts from white objects and close them
        tmp_f = cv2.morphologyEx(tmp_f, cv2.MORPH_CLOSE, kernel)
        result.append(tmp_f)


def write_xml(struct, output_path, init_i_frame):
     # Now create a new XML structure with frames as the first level
    root = etree.Element("annotations")

    for i_frame, frame in enumerate(struct):
        real_i_frame = i_frame + init_i_frame
        frame_element = etree.SubElement(root, "frame", number=real_i_frame)

        for (xtl, ytl, xbr, ybr) in frame:
            # Create a new <box> element and copy its attributes
            box_element = etree.SubElement(frame_element, "box", frame=str(real_i_frame), 
                                          xtl=str(xtl), ytl=str(ytl),
                                          xbr=str(xbr), ybr=str(ybr))

    # Save the new XML file
    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return 

def box_generator(mask, output_path, i_frame):
    result = []
    for frame in mask:
        bb_frame_result = []
        binary_image = 255*frame
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
        for i in range(1, num_labels):  # Start from 1 to skip the background label (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bb = (x, y, x + w, y + h)
            bb_frame_result.append(bb)
        result.append(bb_frame_result)
    write_xml(result, output_path, i_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./fixed_gaussian_boxes.xml", help='Output xml')
    parser.add_argument('--path_video', required=False, default="../AICity_data/train/S03/c010/vdo.avi", help='Input video')
    
    parser.add_argument('--alpha', required=False, default=2, help='Alpha factor')
    parser.add_argument('--video_init', required=False, default=0.25, help='Initial percentage of the video to use.')

    args = parser.parse_args()
    init_frames, result_frames, split_frame_number = video_loader_splitter(args.path_video, args.video_init)
    mean, std = compute_background_model(init_frames)
    unrefined_mask = compute_background(result_frames, mean, std, args.alpha)# 0 is background/ is foreground
    # video_generator(unrefined_background)
    # mask = refine_frames(unrefined_mask)
    # boxes_frames = box_generator(mask, args.path_xml,split_frame_number)


