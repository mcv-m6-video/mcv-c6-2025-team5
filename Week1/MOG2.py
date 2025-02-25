import cv2
import argparse
import os
import tqdm
import xml.etree.ElementTree as ET
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Side-by-side video with fg mask on left and bounding boxes on right.")
    parser.add_argument('--video', required=True, help="Path to the input video file.")
    parser.add_argument('--output', default="MOG2_his100_var32_0sd_b500", 
                        help="Path to save the side-by-side output video.")
    parser.add_argument('--display', action='store_true', help="If set, display frames in a window.")
    parser.add_argument('--min_area', type=int, default=500, 
                        help="Minimum contour area to consider for bounding boxes.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # for tqdm

    back_sub = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 32, detectShadows = False)

    out_width = 720 * 2
    out_height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(f"{args.output}.mp4", fourcc, fps, (out_width, out_height))

    annotations = {}

    pbar = tqdm.tqdm(total=frame_count, desc="Processing frames")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = back_sub.apply(frame)
        # _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # _, bin_img = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        # contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # kernel = np.ones((3, 3), dtype=np.uint8)
        # opened_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((5, 5), dtype=np.uint8)
        # closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)
        # contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= args.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                x_tl = float(x)
                y_tl = float(y)
                x_br = float(x + w)
                y_br = float(y + h)
                bboxes.append((x_tl, y_tl, x_br, y_br))

        frame_with_boxes = frame.copy()
        for (xtl, ytl, xbr, ybr) in bboxes:
            cv2.rectangle(frame_with_boxes, 
                          (int(xtl), int(ytl)), 
                          (int(xbr), int(ybr)), 
                          (0, 255, 0), 2)

        # fg_mask_bgr = cv2.cvtColor(closed_img, cv2.COLOR_GRAY2BGR)
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        side_by_side = np.hstack((fg_mask_bgr, frame_with_boxes))
        side_by_side = cv2.resize(side_by_side, (720*2,480))
        out_writer.write(side_by_side)

        annotations[frame_index] = bboxes

        if args.display:
            cv2.imshow("Side-by-Side", side_by_side)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_index += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    # XML
    root = ET.Element("annotations")
    for f_idx, boxes in annotations.items():
        frame_element = ET.SubElement(root, "frame", number=str(f_idx))
        for (xtl, ytl, xbr, ybr) in boxes:
            box_attrib = {
                "frame": str(f_idx),
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}",
                "outside": "0",   
                "occluded": "0",  
                "keyframe": "1" 
            }
            ET.SubElement(frame_element, "box", box_attrib)

    tree = ET.ElementTree(root)
    tree.write(f"{args.output}.xml", encoding="UTF-8", xml_declaration=True)

    print(f"Processing complete. Side-by-side video saved to '{args.output}'.")
    print(f"Bounding box annotations saved to '{args.output}.xml'.")

if __name__ == "__main__":
    main()
