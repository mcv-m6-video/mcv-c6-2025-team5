import cv2
from bs4 import BeautifulSoup
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
from pycocotools.cocoeval import COCOeval

import argparse
import sys
import glob
import os
import json


def load_pred(pred_path, output_json_path):
    json_list = []
    with open(pred_path, 'r') as file:
        for line in file:
            fields = line.strip().split(",")
            frame_id = int(fields[0]) - 1
            
            x = float(fields[2])
            y = float(fields[3])
            w = float(fields[4])
            h = float(fields[5])
            bbox = [x, y, w, h]
            score = float(fields[6])
            json_list.append({ "image_id": frame_id, "category_id": 1, "bbox": bbox, "score": score })

    
    with open(output_json_path, "w") as json_file:
        json.dump(json_list, json_file, indent=4)
    
    return output_json_path

def compute_metric(gt_json, pred_json, method_name):
    print(f"Metric result of predictions of {method_name}")
    COCO_gt = COCO(gt_json)
    coco_pred = COCO_gt.loadRes(pred_json)

    coco_eval = COCOeval(COCO_gt, coco_pred, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize() 

def parse_pairs(pairs_str):
    pairs = []
    for pair in pairs_str:
        # Remove parentheses and split by comma
        description, path = pair.strip("()").split(",", 1)  # Split only on the first comma
        pairs.append((description.strip(), path.strip()))  # Strip extra spaces
    return pairs

def load_json(path_json):
    try:
        with open(path_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{path_json}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in file '{path_json}'.")
    return None

def load_structure(path_json, gt=False):
    json_dict=load_json(path_json)
    struct = dict()
    if gt:
        list_anot = json_dict["annotations"]
    else:
        list_anot = json_dict
    for anot in list_anot:
        frame_id = anot["image_id"]
        xtl,ytl,w,h = anot["bbox"]
        xbr = xtl + w
        ybr = ytl + h
        bbox = [xtl, ytl, xbr, ybr]
        if not frame_id in struct:
            struct[frame_id] = []
        
        if not gt:
            score = anot["score"]
            struct[frame_id].append((bbox,score))
        else:
            struct[frame_id].append(bbox)
    return struct

def video_creation(gt_bb, result_bb, in_video_path, out_video_path='output_video_with_boxes.mp4'):
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

        if frame_number in result_bb:
            for (box, score) in result_bb[frame_number]:
                xtl, ytl, xbr, ybr = box
                if score >= 0.5:
                    cv2.rectangle(frame, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (255,0,0), 2)  # Blue box with thickness 2
                    label = f"{score:.2f}"
                    cv2.putText(frame, label, (int(xtl), int(ytl) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


        # Write the frame with boxes to the output video
        output_video.write(frame)
        
        frame_number += 1

    # Release the video capture and writer objects
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("Video with bounding boxes saved successfully!")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=False, default="./week2_anot.json", help='GT to compare.')
    parser.add_argument('--path_video', required=False, default="./AICity_data/train/S03/c010/vdo.avi", help='Input video')
    parser.add_argument("--pairs", required=True, type=str, nargs="+", help="List of (method_name, json_path) pairs")
    parser.add_argument('--metric', action='store_true', help="Enable measuring predictions")
    parser.add_argument('--draw', action='store_true', help="Enable drawing predictions")
    args = parser.parse_args()

    predictions = parse_pairs(args.pairs)
    
    if args.metric or not args.draw:
        for method_name, pred_path in predictions:
            compute_metric(args.gt_path, pred_path, method_name)
    
    if args.draw:
        gt_bb = load_structure(args.gt_path, gt=True)
        for method_name, pred_path in predictions:
            pred = load_structure(pred_path)
            video_creation(gt_bb, pred, args.path_video, out_video_path=f"boxes_{method_name}.mp4")
