# mcv-c6-2025-team5 Week 2

## Object detection
### Off the shelf inferences
* To execute the FAST R CNN inference execute this command:
```
```
### Off the shelf metric computation

* Assuming all the inferences are present on ./predictions/ in the form of JSON with the COCO format.
* Execute the following command for computing the metrics of the predictions:
```
python evaluate_method.py --pairs "(mask_rcnn, ./predictions/mask_rcnn_prediction.json)" "(ssd512, ./predictions/ssd512_prediction.json)" "(yolo3, ./predictions/yolo3_prediction.json)" "(fastrcnn, ./predictions/fast_offshelf_output_coco_onlyanot.json)"

```
* Execute the following command for computing drawing the resulting bounding boxes in the video:
```
python evaluate_method.py --pairs "(mask_rcnn, ./predictions/mask_rcnn_prediction.json)" "(ssd512, ./predictions/ssd512_prediction.json)" "(yolo3, ./predictions/yolo3_prediction.json)" "(fastrcnn, ./predictions/fast_offshelf_output_coco_onlyanot.json)" --draw

```
### Fine tunning

## Object tracking

* Everything can be executed and seen in the [notebook](./WEEK2_Tracking.ipynb)