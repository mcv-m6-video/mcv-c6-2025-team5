import cv2
import numpy as np

from bs4 import BeautifulSoup
from lxml import etree

import argparse
import sys

def box_generator(binary_image, min_area=500):
    bb_frame_result = []
    #binary_image = 255*mask
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            x_tl = float(x)
            y_tl = float(y)
            x_br = float(x + w)
            y_br = float(y + h)
            bb_frame_result.append((x_tl, y_tl, x_br, y_br))
    return bb_frame_result

def write_xml(struct, output_path):
    # Now create a new XML structure with frames as the first level
    root = etree.Element("annotations")

    for i_frame, frame in enumerate(struct):
        frame_element = etree.SubElement(root, "frame", number=str(i_frame))

        for (xtl, ytl, xbr, ybr) in frame:
            # Create a new <box> element and copy its attributes
            box_element = etree.SubElement(frame_element, "box", frame=str(i_frame), 
                                          xtl=str(xtl), ytl=str(ytl),
                                          xbr=str(xbr), ybr=str(ybr))

    # Save the new XML file
    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8") 

def lsbp(path_video, out_video_path="./LSBP.mp4", out_xml="./LSBP_boxes.xml", make_output_video=True, display=False, bb_generation=True):
    # Fine-tuned LSBP background subtractor for street traffic detection
    print("Init script...")
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP(
        mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE,
        nSamples=50,             
        LSBPRadius=16,           
        Tlower=6.0,             
        Tupper=55.0,            
        Tinc=1.5,                
        Tdec=0.1,                
        Rscale=8.0,
        Rincdec=0.01,
        noiseRemovalThresholdFacBG=0.0008,
        noiseRemovalThresholdFacFG=0.0012,
        LSBPthreshold=10,
        minCount=4)
# mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE,
        # nSamples=50,             
        # LSBPRadius=16,           
        # Tlower=4.0,             
        # Tupper=40.0,            
        # Tinc=1.5,                
        # Tdec=0.1,                
        # Rscale=15.0,
        # Rincdec=0.01,
        # noiseRemovalThresholdFacBG=0.0008,
        # noiseRemovalThresholdFacFG=0.0012,
        # LSBPthreshold=10,
        # minCount=3)
    cap = cv2.VideoCapture(path_video)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    it = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Beggining processing")
    mask_video = []
    box_struct = []
    
    for i, frame in enumerate(frames):
        print(f"processing frame {i}")
        #blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # Apply LSBP background subtraction
        fg_mask = bg_subtractor.apply(frame) 
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        if make_output_video:
            mask_video.append(fg_mask)
        if bb_generation:
            box_struct.append(box_generator(fg_mask)) # assume black background and white foreground 0 and 255 respectively
        it += 1
        # if it > 100:
        #     break
    if make_output_video:
        output_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=False)
        for i,fg_mask in enumerate(mask_video):
            #cv2.imwrite(f"./frames/frame_{i}.png", fg_mask)
            output_video.write(fg_mask)
        output_video.release()

    if bb_generation:
        write_xml(box_struct, out_xml)


    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./LSBP_boxes.xml", help='Output xml')
    parser.add_argument('--path_video', required=False, default="./AICity_data/train/S03/c010/vdo.avi", help='Input video')
    args = parser.parse_args()
    print("LSBP init")
    lsbp(args.path_video, out_xml=args.path_xml)

