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
TRAIN_JSON = os.path.join(DATASET_PATH, "gt_train_strategy_A.json")
VAL_JSON = os.path.join(DATASET_PATH, "gt_eval_strategy_A.json")
IMAGES_PATH = os.path.join(DATASET_PATH, "images")

# ----- REGISTER DATASET -----
register_coco_instances(DATASET_NAME + "_train", {}, TRAIN_JSON, IMAGES_PATH)
register_coco_instances(DATASET_NAME + "_val", {}, VAL_JSON, IMAGES_PATH)

# ----- CONFIGURE MODEL -----
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (DATASET_NAME + "_train",)
cfg.DATASETS.TEST = (DATASET_NAME + "_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Load COCO-pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")


cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Training settings
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 400
cfg.SOLVER.STEPS = []

# Set batch size for evaluation
cfg.TEST.EVAL_PERIOD = 400  # Evaluate every 5 iterations

# ----- TRAINER CONFIGURATION -----
# Set up the trainer (which will automatically build the model)
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
evaluator = COCOEvaluator(DATASET_NAME + "_val", cfg, False, output_dir="./output/")

# ----- TRAIN MODEL -----
trainer.train()
trainer.test(cfg, model, evaluators=[evaluator])

# ----- SAVE MODEL -----
torch.save(cfg, "faster_rcnn_finetuned.pth")
print("Model saved!")
