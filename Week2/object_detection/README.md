# mcv-c6-2025-team5 Week 2

## Object detection
### Off the shelf inferences
### Fine tunning
### Scores and bounding boxes

* Run off the shelf fast r cnn
* Run fine tunning, with k fold strategy

* Draw boxes 

## Object tracking

* Everything can be executed and seen in the [notebook](./WEEK_1_colab_version.ipynb)

## RUN Adaptative gaussian
* Having the AICityData on the same folder as the scripts

* Execute adaptative gaussian:
```
python adaptive_gaussian_background_model.py --path_video ./AICity_data/train/S03/c010/vdo.avi --alpha 9 --ro 0.6
```

## RUN MOG2 and LSBP
* Having the AICityData on the same folder as the scripts

* Execute MOG2:
```
python MOG2.py --video ./AICity_data/train/S03/c010/vdo.avi
```

* Execute LBSP:

```
python LSBP.py --path_video ./AICity_data/train/S03/c010/vdo.avi
```


## Run DL method (DeepMCBM)

* Go to the folder: 

```
cd DeepMCBM
```

* Install env

```
conda env create -f DeepMCBM_env.yml
```

* Prepare dataset:

```
python ../vid_to_frame.py --video path/to/vdo.avi --output path/to/output/vdo
``` 

* Train:

```
/mnt/home/.conda/envs/mcbm/bin/python src/DeepMCBM.py --parent_dir .path/to/output/vdo --dir frames_vdo_train25
```

* Predict (We also pushed the pretrained to repo, so you can run inference directly without training):
```
/mnt/home/.conda/envs/mcbm/bin/python src/DeepMCBM.py --parent_dir .path/to/output/vdo --dir frames_vdo_eval75 --no_train_STN --no_train_BMN 
```

XML result file: `DeepMCBM/output/frames_vdo/my_run/annotations.xml`
video result file: `DeepMCBM/output/frames_vdo/my_run/side_by_side.mp4`

## Run metrics and draw boxes
* Suppose fix gaussiand, adaptative gaussian, LSBP, MOG2  and DL method are execute, with the respective xml files.

* Compute the metrics:
```
python metric_computation.py --pairs  "(Gauss_adapt, ./adaptive_gaussian_preds.xml)" "(Gauss_static, static_gaussian_preds.xml)" "(MOG2, ./MOG2_his100_var32_0sd_b500.xml)" "(LSBP,LSBP_boxes.xml)" "(DL, ./annotations.xml)"
```

* Generate bounding boxes videos:
```
python draw_boxes.py --pairs "(Gauss_adapt, ./adaptive_gaussian_preds.xml)" "(Gauss_static, static_gaussian_preds.xml)" "(MOG2, ./MOG2_his100_var32_0sd_b500.xml)" "(LSBP,LSBP_boxes.xml)" "(DL, ./annotations.xml)"
```
