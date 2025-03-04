from bs4 import BeautifulSoup
from lxml import etree

import numpy as np
import argparse
import pickle
import json

def generate_basic_gt(struct, w, h, path="week2_gt.json"):
    json_dict = {}
    json_dict["images"] = []
    json_dict["annotations"] = []
    json_dict["categories"] = [{ "id": 1, "name": "car" }, { "id": 2, "name": "bike" }]

    counter = 0
    max_frame = 0
    set_frames = set()
    for frame_numb in (sorted(struct.keys())):
        json_dict["images"].append({ "id": frame_numb,"file_name":str(frame_numb)+".jpg", "width": w, "height": h })
        if frame_numb > max_frame:
            max_frame = frame_numb
        set_frames.add(frame_numb)
        for (bb_x, bb_y, bb_width, bb_height), label in (struct[frame_numb]):
            if label == "car":
                category_id = 1
            elif label == "bike":
                category_id = 2
            else:
                print(f"ERROR WRONG LABEL: {label}")
                sys.exit(1)
            if category_id == 1:
                bbox = [bb_x, bb_y, bb_width, bb_height]
                json_dict["annotations"].append({ "id": counter, "image_id": frame_numb, "category_id": category_id, "bbox": bbox, "area": bb_width*bb_height, "iscrowd": 0 })
                counter += 1

    for i in range(max_frame+1):
        if not i in set_frames:
            json_dict["images"].append({ "id": i, "width": w, "height": h })

    with open(path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="./ai_challenge_s03_c010-full_annotation.xml", help='Input xml')
    parser.add_argument('--new_json', required=False, default="./week2_anot.json", help='Json output')


    result_struct = {}
    # Parse the arguments and call the appropriate function
    args = parser.parse_args()
    with open(args.path_xml, "r", encoding="utf-8") as file:
        xml_content = file.read()
    
    soup = BeautifulSoup(xml_content, "xml")
    tracks = soup.find_all("track")
    width = int(soup.find("width").text)
    height = int(soup.find("height").text)
    
    for track in tracks:
        track_id = track.get('id')
        track_label = track.get('label')
        boxes = track.find_all('box')
        for box in boxes:
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            outside = box.get('outside')
            occluded = box.get('occluded')
            keyframe = box.get('keyframe')

            bb_x = xtl
            bb_y = ytl
            bb_width = xbr - xtl
            bb_height = ybr - ytl
            bbox = [bb_x, bb_y, bb_width, bb_height]
            if not frame in result_struct:
                result_struct[frame] = []
            result_struct[frame].append((bbox,track_label))
    
    generate_basic_gt(result_struct, width, height, path = args.new_json)