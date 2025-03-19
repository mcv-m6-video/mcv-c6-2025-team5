import os
import argparse
import pandas as pd
import json

from collections import defaultdict
import numpy as np

from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity

def iou(box_a, box_b):
    """
    Computes the Intersection-over-Union (IoU) of two boxes.
    Each box is in the format [x, y, w, h].
    """
    # Convert [x, y, w, h] to (xmin, ymin, xmax, ymax)
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    # Intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    # Areas of each box
    area_a = box_a[2] * box_a[3]  # w*h
    area_b = box_b[2] * box_b[3]

    union_area = area_a + area_b - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def union_box(box_a, box_b):
    """
    Returns the bounding-box union of the two boxes
    (the minimal rectangle that encloses both).
    Each box is in the format [x, y, w, h].
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    union_x1 = min(ax1, bx1)
    union_y1 = min(ay1, by1)
    union_x2 = max(ax2, bx2)
    union_y2 = max(ay2, by2)

    return [union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1]


def union_of_boxes(list_of_boxes):
    """
    Given a list of [x, y, w, h] boxes, returns the bounding box that encloses them all.
    """
    if not list_of_boxes:
        return None

    # Initialize x1, y1 with a large value, x2, y2 with a small value
    x1 = float('inf')
    y1 = float('inf')
    x2 = float('-inf')
    y2 = float('-inf')

    for (x, y, w, h) in list_of_boxes:
        # Convert [x, y, w, h] into corners
        bx1, by1 = x, y
        bx2, by2 = x + w, y + h

        # Update union coordinates
        x1 = min(x1, bx1)
        y1 = min(y1, by1)
        x2 = max(x2, bx2)
        y2 = max(y2, by2)

    # Convert corners back to [x, y, w, h]
    return [x1, y1, x2 - x1, y2 - y1]


def merge_overlapping_boxes(bboxes, iou_threshold=0.5):
    """
    Given a list of bounding boxes [ [x, y, w, h], ... ],
    merges any two boxes whose IoU > iou_threshold into their union.
    Repeats until no further merges are found.
    Returns the merged list of boxes.
    """
    merged = True
    boxes = bboxes[:]

    # Keep merging until no more merges happen
    while merged:
        merged = False
        new_boxes = []
        while boxes:
            current_box = boxes.pop()
            # Try to merge current_box with one of the boxes already in new_boxes
            for i, nb in enumerate(new_boxes):
                if iou(current_box['bbox'], nb['bbox']) > iou_threshold:
                    # Merge them
                    merged_box = union_box(current_box['bbox'], nb['bbox'])
                    merged_conf = max(current_box['conf'], nb['conf'])
                    # Replace the box in new_boxes with the merged box
                    new_boxes[i] = {'bbox': merged_box, 'conf': merged_conf}
                    merged = True
                    break
            else:
                # If we never broke, it means no merge happened; keep current_box
                new_boxes.append(current_box)
        boxes = new_boxes

    return boxes

def parse_tracking_file(filepath):
    """
    Reads the detection text file and returns data grouped by frame.
    Each element in the returned list corresponds to a single frame,
    which itself is a list of dictionaries.
    Each dictionary has keys: 'bbox' -> [left, top, width, height], 'conf' -> conf_value

    :param filepath: Path to the input text file.
    """

    # Using a dictionary to accumulate detections by frame number:
    frames_dict = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            # Strip and skip any empty lines
            line = line.strip()
            if not line:
                continue

            # Split line into fields
            fields = line.split(',')
            # fields are expected as: frame, track_id, left, top, width, height, conf, -1, -1, -1

            frame = int(fields[0].strip())
            track_id = int(fields[1].strip())
            left  = float(fields[2].strip())
            top   = float(fields[3].strip())
            width = float(fields[4].strip())
            height= float(fields[5].strip())
            conf  = float(fields[6].strip())

            # Construct detection dictionary
            detection = {
                'bbox': [left, top, width, height],
                'conf': conf,
                'track_id': track_id
            }

            # Append the detection to the corresponding frame
            frames_dict[frame].append(detection)

    return frames_dict


def save_tracking_data(filepath, tracking_data):
    """
    Saves tracking data to a file in the format:
      frame,id,left,top,width,height,conf,-1,-1,-1

    :param filepath: Path to the output text file.
    :param tracking_data: A list of frames (list),
                          where each frame is a list of dictionaries.
                          Each dictionary has keys:
                            {
                                'id': <integer ID>,
                                'bbox': [left, top, width, height],
                                'conf': <float confidence>
                            }
    """
    with open(filepath, 'w') as f:
        # 'frame_idx' will start from 1, but adjust if your frames are 0-based
        for frame_idx, detections in enumerate(tracking_data, start=1):
            for det in detections:
                box_id = det['track_id']
                left, top, width, height = det['bbox']
                conf = det['conf']
                # Write one line per detection
                line = f"{frame_idx},{box_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1"
                f.write(line + "\n")

def read_csv_file(path):
    result = dict()
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        frame = str(row['frame'])
        bbox_x = int(row['bbox_topleft_x'])
        bbox_y = int(row['bbox_topleft_y'])
        bbox_width = int(row['bbox_width'])
        bbox_height = int(row['bbox_height'])
        track_id = int(row['track_id'])
        if not frame in result:
            result[frame] = []
        
        result[frame].append({
              "track_id": track_id,
              "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
              "confidence": 1.0
        })

    return result


def read_preds(pred_pairs):
    json = {"videos": {"seq":None}}
    cams = dict()
    for cam_name, path in pred_pairs:
        dict_key_frames_cam = read_csv_file(path)
        cams[cam_name] = {"frames":dict_key_frames_cam}
    json["videos"]["seq"] = cams

    return json


def read_gt(gt_pairs):
    json = {"videos": {"seq":None}}
    cams = dict()
    for cam_name, path in pred_pairs:
        dict_key_frames_cam = read_csv_file(path)
        cams[cam_name] = {"frames":dict_key_frames_cam}
    json["videos"]["seq"] = cams

    return json

def calculate_metrics(tracker_data, gt_data):
    # build mapping dict for tracker
    unique_tracker_ids_tr = set()
    for frame, dets in tracker_data.items():
        for det in dets:
            if 'track_id' not in det.keys():
                print(frame)
            unique_tracker_ids_tr.add(det['track_id'])
    unique_tracker_ids_tr = sorted(list(unique_tracker_ids_tr))
    tracker_id_mapping_tr = {old_id: new_id for new_id, old_id in enumerate(unique_tracker_ids_tr)}
    
    # build mapping dict for gt
    unique_tracker_ids_gt = set()
    for frame, dets in gt_data.items():
        for det in dets:
            unique_tracker_ids_gt.add(det['track_id'])
    unique_tracker_ids_gt = sorted(list(unique_tracker_ids_gt))
    tracker_id_mapping_gt = {old_id: new_id for new_id, old_id in enumerate(unique_tracker_ids_gt)}
    
    all_frames = sorted(set(gt_data.keys()).union(tracker_data.keys()))
    
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    total_tracker_dets = 0
    total_gt_dets = 0
    
    for frame in all_frames:
        if frame in gt_data:
            gt_dets = gt_data[frame]
            # remap track IDs for gt
            gt_ids = np.array([tracker_id_mapping_gt[det['track_id']] for det in gt_dets])
            if frame in tracker_data:
                tr_dets = tracker_data[frame]
                # remap track IDs for tr
                tr_ids = np.array([tracker_id_mapping_tr[det['track_id']] for det in tr_dets])
            else:
                tr_dets = [{'bbox': [0, 0, 0, 0],
                  'category_id': 0,
                  'track_id': 0,
                  'conf': 0}]
                tr_ids =  np.array([det['track_id'] for det in tr_dets])
            
            total_gt_dets += len(gt_dets)
            total_tracker_dets += len(tr_dets)
            
            if len(gt_dets) > 0 and len(tr_dets) > 0:
                sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)
                for i, gt in enumerate(gt_dets):
                    for j, tr in enumerate(tr_dets):
                        sim_matrix[i, j] = iou(gt['bbox'], tr['bbox'])
            else:
                sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)
            
            gt_ids_list.append(gt_ids)
            # gt_ids_list.append(np.array([int(a['track_id']) for a in gt_dets]))
            tracker_ids_list.append(tr_ids)
            # tracker_ids_list.append(np.array([int(a['track_id']) for a in tr_dets]))
            similarity_scores_list.append(sim_matrix)
    
    
    num_gt_ids = len(unique_tracker_ids_gt)
    num_tracker_ids = len(unique_tracker_ids_tr)
    
    # data dictionary for HOTA
    data = {
        'num_tracker_dets': total_tracker_dets,
        'num_gt_dets': total_gt_dets,
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_tracker_ids,
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_scores_list
    }
    hota_metric = HOTA()
    identity_metric = Identity()
    result_hota = hota_metric.eval_sequence(data)
    result_identity = identity_metric.eval_sequence(data)
    return result_hota, result_identity

def calculate_metrics_taking_only_GTobject_into_account(tracker_data, gt_data, iou_threshold=0.01):
    """
    Evaluates metrics using only GT objects. Instead of relying on matching IDs (since
    GT and tracker IDs differ), we filter tracker detections per frame. We retain only
    tracker detections that have an IoU >= iou_threshold with at least one GT box.
    In frames where no valid tracker detections are found, a dummy detection is inserted
    to avoid failure in the HOTA function.
    """
    # Build mapping dict for GT track IDs (from GT detections only)
    unique_gt_ids = set()
    for frame, dets in gt_data.items():
        for det in dets:
            unique_gt_ids.add(det['track_id'])
    unique_gt_ids = sorted(list(unique_gt_ids))
    gt_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_gt_ids)}

    # Filter tracker detections per frame using IoU threshold, only for frames present in GT
    filtered_tracker_data = {}
    for frame in gt_data.keys():
        valid_tr_dets = []
        if frame in tracker_data:
            for tr in tracker_data[frame]:
                # Compute IoU with each GT detection in this frame
                ious = [iou(gt['bbox'], tr['bbox']) for gt in gt_data[frame]]
                if ious and max(ious) >= iou_threshold:
                    valid_tr_dets.append(tr)
        # If no valid tracker detection exists, insert the dummy detection
        if not valid_tr_dets:
            valid_tr_dets = [{
                'bbox': [0, 0, 0, 0],
                'category_id': 0,
                'track_id': 0,
                'conf': 0
            }]
        filtered_tracker_data[frame] = valid_tr_dets

    # Build mapping for tracker track IDs from the filtered data
    unique_tracker_ids = set()
    for frame in gt_data.keys():
        for det in filtered_tracker_data[frame]:
            unique_tracker_ids.add(det['track_id'])
    unique_tracker_ids = sorted(list(unique_tracker_ids))
    tracker_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracker_ids)}

    # Evaluate on frames that exist in GT
    all_frames = sorted(gt_data.keys())
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    total_tracker_dets = 0
    total_gt_dets = 0

    for frame in all_frames:
        gt_dets = gt_data[frame]
        # Remap GT track IDs
        gt_ids = np.array([gt_id_mapping[det['track_id']] for det in gt_dets])
        tr_dets = filtered_tracker_data.get(frame, [{
            'bbox': [0, 0, 0, 0],
            'category_id': 0,
            'track_id': 0,
            'conf': 0
        }])
        tr_ids = np.array([tracker_id_mapping[det['track_id']] for det in tr_dets])
        # print(gt_ids, tr_ids)
        total_gt_dets += len(gt_dets)
        total_tracker_dets += len(tr_dets)

        # Build similarity matrix using IoU for the current frame
        if len(gt_dets) > 0 and len(tr_dets) > 0:
            sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)
            for i, gt in enumerate(gt_dets):
                for j, tr in enumerate(tr_dets):
                    sim_matrix[i, j] = iou(gt['bbox'], tr['bbox'])
        else:
            sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)

        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tr_ids)
        similarity_scores_list.append(sim_matrix)

    num_gt_ids = len(unique_gt_ids)
    num_tracker_ids = len(unique_tracker_ids)

    # Create the data dictionary expected by the HOTA evaluation metric
    data = {
        'num_tracker_dets': total_tracker_dets,
        'num_gt_dets': total_gt_dets,
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_tracker_ids,
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_scores_list
    }

    hota_metric = HOTA()
    identity_metric = Identity()  # Assuming your Identity metric is defined similarly
    result_hota = hota_metric.eval_sequence(data)
    result_identity = identity_metric.eval_sequence(data)
    return result_hota, result_identity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="output/cityflow_s01", required=False, help="Path for predict")
    parser.add_argument("--gt", default="/home/marco/Downloads/aic19-track1-mtmc-train/train/S01/", required=False, help="Path for gt")
    args = parser.parse_args()

    # List all subfolders in the base directory
    subfolders = os.listdir(args.pred)

    # List to store pairs (subfolder, mtmc.csv path)
    pred_pairs = []

    # Loop through each subfolder
    for subfolder in subfolders:
        # Define the path to the mtmc.csv file in the subfolder
        csv_path = os.path.join(args.pred, subfolder, 'mtmc.txt')

        # Check if mtmc.csv exists in the current subfolder
        if os.path.exists(csv_path):
            # Append the pair (subfolder name, path to mtmc.csv) to the list
            pred_pairs.append((subfolder, csv_path))
        else:
            print(f"mtmc.txt not found in {subfolder}")

    # Display the list of pairs
    print(sorted(pred_pairs))
    print("--------")
    # List to store pairs (subfolder_name, gt.txt path)
    gt_pairs = []

    # Loop through each subfolder (c0, c1, c2, ...)
    for subfolder in os.listdir(args.gt):
        # Ensure we're working with the correct subfolder c0, c1, ...
        if subfolder.startswith('c'):
            # Define the path to the gt.txt file
            gt_path = os.path.join(args.gt, subfolder, 'gt', 'gt.txt')

            # Check if the gt.txt file exists
            if os.path.exists(gt_path):
                # Append the pair (subfolder name, path to gt.txt) to the list
                gt_pairs.append((subfolder, gt_path))
            else:
                print(f"gt.txt not found in {subfolder}")

    # Display the list of pairs
    print(sorted(gt_pairs))
    avg1 = []
    avg2 = []
    for (name_pred, path_pred), (name_gt, path_gt) in zip(sorted(pred_pairs),sorted(gt_pairs)):
        tracking = parse_tracking_file(path_pred)
        annotations = parse_tracking_file(path_gt)
        mt2 = calculate_metrics_taking_only_GTobject_into_account(tracking, annotations)
        
        print(f"Name pred: {name_pred}, gt of {name_gt}, HOTA ", mt2[0]["HOTA(0)"], "and IDF1:", mt2[1]["IDF1"])
        # print(mt2)
        avg1.append(mt2[0]["HOTA(0)"])
        avg2.append(mt2[1]["IDF1"])

    print(f"HOTA: {np.mean(np.array(avg1))*100} %")
    print(f"IDF1: {np.mean(np.array(avg2))*100} %")