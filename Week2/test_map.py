from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def load_predictions(pred_json):
    """
    Extracts and reformats the prediction data from the custom structure into COCO's expected format.
    
    Args:
        pred_json (str): Path to the predicted JSON file.

    Returns:
        list: List of predictions in the correct COCO format.
    """
    with open(pred_json, 'r') as f:
        data = json.load(f)
    
    # Check the structure of predictions (for debugging)
    print("Structure of predictions data:", data.get('annotations', None))

    predictions = []
    
    # Assuming predictions are inside a list under 'predictions' key
    for prediction in data.get('annotations', []):
        # Check the type of 'prediction' to debug the issue
        print("Type of prediction:", type(prediction))
        if isinstance(prediction, dict):
            coco_pred = {
                'image_id': prediction['image_id'],
                'category_id': prediction['category_id'],
                'bbox': prediction['bbox'],  # [x, y, width, height]
                'score': prediction['score']
            }
            predictions.append(coco_pred)
        else:
            print("Skipping invalid prediction:", prediction)

    return predictions

def evaluate_coco(gt_json, pred_json):
    """
    Evaluates the mAP between ground truth and predicted COCO JSON files.

    Args:
        gt_json (str): Path to the ground truth COCO JSON file.
        pred_json (str): Path to the predicted COCO JSON file.

    Returns:
        dict: COCO evaluation results including mAP@IoU=[0.50:0.95], mAP@50, and mAP@75.
    """
    # Load ground truth
    coco_gt = COCO(gt_json)
    
    # Load and reformat predictions
    coco_dt = coco_gt.loadRes(load_predictions(pred_json))

    # Initialize COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract key metrics
    results = {
        "mAP_50:95": coco_eval.stats[0],  # mAP@[0.50:0.95]
        "mAP_50": coco_eval.stats[1],     # mAP@50
        "mAP_75": coco_eval.stats[2]      # mAP@75
    }
    
    return results

# Paths to JSON files
gt_json = "week2_anot.json"   # Path to COCO ground truth
pred_json = "output_predictions.json"  # Path to COCO predictions

# Run evaluation
results = evaluate_coco(gt_json, pred_json)
print("\nEvaluation Results:", results)
