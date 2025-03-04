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

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs", required=True, type=str, nargs="+", help="List of (method_name, txt_path) pairs")
    args = parser.parse_args()

    #json_gt = load_json(args.gt_path)
    txt_paths = parse_pairs(args.pairs)

    for method_name, path in txt_paths:
        json_path = load_pred(path, method_name+"_prediction.json")