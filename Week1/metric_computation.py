import cv2
from bs4 import BeautifulSoup
import numpy as np
import argparse
import random
import json
import numpy as np
import sys
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WIDTH = 0
HEIGHT = 0

def AP_50_metric(gt, json_pred, K = 10):

    # Group predictions by image_id
    pred_by_image = {}
    for pred in json_pred:
        image_id = pred["image_id"]
        if image_id not in pred_by_image:
            pred_by_image[image_id] = []
        pred_by_image[image_id].append(pred)

    # Store AP50 per image
    ap_per_image = {}

    # Iterate over each image
    for image_id, preds in pred_by_image.items():
        ap_values = []  # Store AP50 for this image

        for _ in range(K):
            # Shuffle predictions
            random.shuffle(preds)

            # Assign decreasing scores
            for idx, pred in enumerate(preds):
                pred["score"] = 1.0 - (idx / len(preds))

            # Save shuffled predictions for this image
            temp_pred_path = f"shuffled_predictions.json"
            with open(temp_pred_path, "w") as f:
                json.dump(preds, f)


            #Ugly workaround to avoid spam cli
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Load shuffled predictions into COCO
            coco_pred = gt.loadRes(temp_pred_path)

            # Run COCO evaluation
            coco_eval = COCOeval(gt, coco_pred, "bbox")
            coco_eval.params.imgIds = [image_id]  # Evaluate only this image
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()  # This won't print anything

            # Restore stdout
            sys.stdout = old_stdout

            # Store AP50 result (index 1 in stats array corresponds to AP50)
            ap_values.append(coco_eval.stats[1])

        # Compute mean AP50 for this image
        ap_per_image[image_id] = np.mean(ap_values)

    # Compute final AP50 for the video (average of all image AP50s)
    video_ap50 = np.mean(list(ap_per_image.values()))
    return video_ap50

def read_xml(path_xml, gt=False):
    global WIDTH
    global HEIGHT
    # Read the XAML file
    with open(path_xml, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Parse the XAML content
    soup = BeautifulSoup(xml_content, "xml")

    # Dictionary to store boxes per frame
    frames_data = {}

    if gt:
        meta_tag = soup.find("meta")
        WIDTH = int(meta_tag["width"])
        HEIGHT = int(meta_tag["height"])
    
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

def parse_pairs(pairs_str):
    pairs = []
    for pair in pairs_str:
        # Remove parentheses and split by comma
        description, path = pair.strip("()").split(",", 1)  # Split only on the first comma
        pairs.append((description.strip(), path.strip()))  # Strip extra spaces
    return pairs

def gt_generate_json(gt_bb, path="./gt.json"):
    json_dict = {}
    json_dict["images"] = []
    json_dict["annotations"] = []
    json_dict["categories"] = [{ "id": 1, "name": "foreground" }]
    counter = 0
    max_frame = 0
    set_frames = set()
    for frame_numb in (sorted(gt_bb.keys())):
        json_dict["images"].append({ "id": frame_numb, "width": WIDTH, "height": HEIGHT })
        if frame_numb > max_frame:
            max_frame = frame_numb
        set_frames.add(frame_numb)
        # print(f"Frame: {frame_numb}")
        for (xtl, ytl, xbr, ybr) in (gt_bb[frame_numb]):
            # print(f"Iterator bb:{i}")
            bb_x = xtl
            bb_y = ytl
            bb_width = xbr - xtl
            bb_height = ybr - ytl
            bbox = [bb_x, bb_y, bb_width, bb_height]
            json_dict["annotations"].append({ "id": counter, "image_id": frame_numb, "category_id": 1, "bbox": bbox, "area": bb_width*bb_height, "iscrowd": 0 })
            counter += 1

    for i in range(max_frame+1):
        if not i in set_frames:
            json_dict["images"].append({ "id": i, "width": WIDTH, "height": HEIGHT })

    with open(path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return path

def pred_generate_json(method_name, pred_bb):
    json_list = []
    for frame_numb in sorted(pred_bb.keys()):
        for (xtl, ytl, xbr, ybr) in (pred_bb[frame_numb]):
            bb_x = xtl
            bb_y = ytl
            bb_width = xbr - xtl
            bb_height = ybr - ytl
            bbox = [bb_x, bb_y, bb_width, bb_height]
            json_list.append({ "image_id": frame_numb, "category_id": 1, "bbox": bbox, "score": -1 }) # scores need to be modified at each shufle
     
    with open(f"pred_{method_name}.json", "w") as json_file:
        json.dump(json_list, json_file, indent=4)
    
    return json_list

def metric(path_gt_xml, preds_pair):
    result = []
    gt_bb = read_xml(path_gt_xml,gt=True)

    gt_path = gt_generate_json(gt_bb)
    COCO_gt = COCO(gt_path)

    for method_name, path in preds_pair:
        pred_xml = read_xml(path)
        pred_json = pred_generate_json(method_name, pred_xml)
        AP_50 = AP_50_metric(COCO_gt, pred_json)
        result.append(AP_50)
        print(f"Ap50 score for method {method_name}: {AP_50}")
    
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./week1_anot.xml", help='Input xml')
    parser.add_argument("--pairs", required=True, type=str, nargs="+", help="List of (method_name, xml_path) pairs")
    args = parser.parse_args()

    name_xml_pairs = parse_pairs(args.pairs)
    print(f"args: {name_xml_pairs}")
    metric(args.path_xml, name_xml_pairs)

    # gt_bb = read_xml(args.path_xml,gt=True)
    # xml_paths = parse_pairs(args.pairs)
    
    # gt_path = gt_generate_json(gt_bb)
    # COCO_gt = COCO(gt_path)

    # for method_name, path in xml_paths:
    #     pred_xml = read_xml(path)
    #     pred_json = pred_generate_json(method_name, pred_xml)
    #     AP_50 = AP_50_metric(COCO_gt, pred_json)
    #     print(f"Ap50 score for method {method_name}: {AP_50}")

    