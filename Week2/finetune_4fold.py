from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os
import torch

# ----- MODIFY THESE VARIABLES -----
DATASET_NAME = "c03_10"
DATASET_PATH = "./dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images") 

for partition in range(4):  
    print(f"\n========== Training on Partition {partition} ==========\n")
    
    TRAIN_JSON = os.path.join(DATASET_PATH, f"gt_train_strategy_B_partition_{partition}.json")
    VAL_JSON = os.path.join(DATASET_PATH, f"gt_eval_strategy_B_partition_{partition}.json")

    # ----- REGISTER DATASET -----
    fold_train_name = f"{DATASET_NAME}_train_partition_{partition}"
    fold_val_name = f"{DATASET_NAME}_val_partition_{partition}"
    register_coco_instances(fold_train_name, {}, TRAIN_JSON, IMAGES_PATH)
    register_coco_instances(fold_val_name, {}, VAL_JSON, IMAGES_PATH)

    # ----- CONFIGURE MODEL -----
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = (fold_train_name,)
    cfg.DATASETS.TEST = (fold_val_name,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "car" class

    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = [] 
    cfg.TEST.EVAL_PERIOD = 400 

    # ----- TRAINER CONFIGURATION -----
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    model = trainer.model


    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)

    # ----- EVALUATOR SETUP -----
    evaluator = COCOEvaluator(fold_val_name, cfg, False, output_dir="./output/")
    
    # ----- TRAIN MODEL -----
    trainer.train()
    trainer.test(cfg, model, evaluators=[evaluator])

    # ----- SAVE MODEL -----
    model_path = f"faster_rcnn_finetuned_partition_{partition}.pth"
    torch.save(cfg, model_path)
    print(f"Model for partition {partition} saved as {model_path}\n")
