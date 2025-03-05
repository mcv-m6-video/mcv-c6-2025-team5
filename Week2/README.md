# mcv-c6-2025-team5 Week 2

## Object detection
### Off the shelf inferences
* To execute the FAST R CNN inference execute this command:
```
python inference.py
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
* Before fine tunning we must execute:
```
python kfold_library.py
```

* To execute the fine tunning of the model, and evaluation, using strategy A:
```
python finetune.py
```

* To execute the fine tunning of the model, and evaluation, using strategy B or C:
```
python finetune_4fold.py
```

## Object tracking

* Everything can be seen in the [notebook](./WEEK2_Tracking.ipynb) or executed in the following [Google Colab](https://colab.research.google.com/drive/11zDIZbtiqjSI3tevftFfwozGUBQi6_kG?hl=es#scrollTo=n76HOGLQmSPc)