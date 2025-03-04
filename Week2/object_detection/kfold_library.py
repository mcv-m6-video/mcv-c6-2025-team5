import json
import numpy as np 

# need a sorted list of frame numbers and the path gt json

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

def save_json(path, json_struct):
    with open(path, "w") as json_file:
        json.dump(json_struct, json_file, indent=4)

def generate_gt(name, set_partition, dict_gt):
    json_dict = {}
    json_dict["images"] = [im for im in dict_gt["images"] if im["id"] in set_partition]
    json_dict["annotations"] = [anot for anot in dict_gt["annotations"]  if anot["image_id"] in set_partition]
    json_dict["categories"] = dict_gt["categories"]
    json_path = name+".json"
    save_json(json_path, json_dict)
    return json_path

def k_fold_partition(frames, K=4):
    np.random.shuffle(frames)
    partitions = np.array_split(frames, K)
    
    return partitions

def generate_strategy_A(list_frames, path_json, K=4):
    json_dict = load_json(path_json)
    partitions = np.array_split(list_frames, K)
    train_frames = set(partitions[0])
    eval_frames = set([y  for x in partitions[1:] for y in x])
    train_gt_json_path = generate_gt("gt_train_strategy_A", train_frames, json_dict)
    eval_gt_json_path = generate_gt("gt_eval_strategy_A", eval_frames, json_dict)

    return [(eval_gt_json_path, eval_frames, train_gt_json_path, train_frames)]

def generate_strategy_B(list_frames, path_json, K=4):
    result = []
    json_dict = load_json(path_json)
    partitions = np.array_split(list_frames, K)
    for i, partition in enumerate(partitions):
        train_frames = set(partitions[i])
        tmp_partitions = partitions[:i] + partitions[i+1:]
        eval_frames = set([y for x in (tmp_partitions) for y in x])
        train_gt_json_path = generate_gt(f"gt_train_strategy_B_partition_{i}", train_frames, json_dict)
        eval_gt_json_path = generate_gt(f"gt_eval_strategy_B_partition_{i}", eval_frames, json_dict)
        result.append((eval_gt_json_path, eval_frames, train_gt_json_path, train_frames))

    return result

def generate_strategy_C(list_frames, path_json, K=4):
    result = []
    json_dict = load_json(path_json)
    partitions = k_fold_partition(list_frames, K)
    for i, partition in enumerate(partitions):
        train_frames = set(partitions[i])
        tmp_partitions = partitions[:i] + partitions[i+1:]
        eval_frames = set([y for x in (tmp_partitions) for y in x])
        train_gt_json_path = generate_gt(f"gt_train_strategy_C_partition_{i}", train_frames, json_dict)
        eval_gt_json_path = generate_gt(f"gt_eval_strategy_C_partition_{i}", eval_frames, json_dict)
        result.append((eval_gt_json_path, eval_frames, train_gt_json_path, train_frames))

    return result

if __name__ == "__main__":
    frames = [x for x in range(2141)]
    strl= "./week2_anot.json"
    generate_strategy_A(frames, strl)
    generate_strategy_B(frames, strl)
    generate_strategy_C(frames, strl)


