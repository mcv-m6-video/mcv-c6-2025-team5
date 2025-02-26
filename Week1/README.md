# mcv-c6-2025-team5

## RUN MOG and LSBP
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
* Suppose LSBP, MOG2  and DL method are execute, with the respective xml files.

* Compute the metrics:
```
python metric_computation.py --pairs "(MOG2, ./MOG2_his100_var32_0sd_b500.xml)" "(LSBP,LSBP_boxes.xml)" "(DL, ./annotations.xml)"
```

* Generate bounding boxes videos:
```
python draw_boxes.py --pairs "(MOG2, ./MOG2_his100_var32_0sd_b500.xml)" "(LSBP,LSBP_boxes.xml)" "(DL, ./annotations.xml)"
```
