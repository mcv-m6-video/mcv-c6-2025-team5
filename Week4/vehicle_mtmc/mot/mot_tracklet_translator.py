from mot.tracklet_processing import load_tracklets,save_tracklets,save_tracklets_csv,save_tracklets_txt
from mot.tracklet import Tracklet

import argparse
import os

def load_txt(path_txt):
    struct = {}
    # Open the file and read line by line
    with open(path_txt, "r") as file:
        for line in file:
            values = line.strip().split(",")  # Remove whitespace and split by comma
            frame_num = int(values[0])
            tracker_id = int(values[1])
            x = float(values[2])
            y = float(values[3]) #  536,1146,581.59,430.52,250.39,208.26,1.00,-1,-1,-1
            w = float(values[4])
            h = float(values[5])
            bbox = [x,y,w,h]
            score = float(values[6])
            if not tracker_id in struct:
                struct[tracker_id] = Tracklet(tracker_id)
            struct[tracker_id].update(frame_num, bbox, score)
    return struct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=False, default="output/translated_results/", help='Path to output the csv,txt and pkl file')
    parser.add_argument('--input_path', required=False, default="../../Week2/our_detection_2.txt", help='Tracklet on our format')
    args = parser.parse_args()
    struct = load_txt(args.input_path)
    array_tracklets = [struct[key] for key in sorted(struct.keys())]
    
    save_tracklets(array_tracklets, os.path.join(args.output_path,"translated_tracklets.pkl"))
    save_tracklets_csv(array_tracklets, os.path.join(args.output_path, "translated_tracklets.csv"))
    save_tracklets_txt(array_tracklets, os.path.join(args.output_path, "translated_tracklets.txt"))


    # d = load_tracklets("./output/cityflow_s01/0_vdo/mot.pkl")
    # print("Type:", type(d))
    # print(d)
    # d[0]