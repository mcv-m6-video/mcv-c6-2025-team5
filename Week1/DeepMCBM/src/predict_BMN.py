import torch
import os
from torch.utils.data import DataLoader, dataloader
from torchvision.transforms import transforms
import neptune.new as neptune
import STN
import STN_Homo
from tqdm import tqdm
import pathlib
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from autoencoder import AutoEncoder
import utils
from utils import update_special_args
from image_dataset import ImageDataset
from BMN import BMN
import args as ARGS
import metrics
from metrics import calc_metric_and_MSE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_side_by_side_and_XML(original_frames_dir, predicted_bg_dir, output_video_path, output_xml_path, min_area=500, out_size=(1440, 480)):
    """
    For each frame in the original frames directory, load the corresponding predicted background image.
    Compute the foreground mask by differencing, find bounding boxes, draw them on the original frame,
    and then create a side-by-side video (left: predicted background, right: original frame with boxes).
    Also, generate an XML file with the bounding boxes.
    """
    # List sorted image filenames (assumes both directories use the same names, e.g. frame000000.png, frame000001.png, etc.)
    orig_files = sorted([f for f in os.listdir(original_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    pred_files = sorted([f for f in os.listdir(predicted_bg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(orig_files) == 0 or len(pred_files) == 0:
        print("No images found in one of the directories.")
        return

    num_frames = min(len(orig_files), len(pred_files))
    
    # Prepare video writer (side-by-side: width doubled)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, 30, out_size)

    # Dictionary to store bounding boxes: frame index -> list of (xtl, ytl, xbr, ybr)
    annotations = {}

    pbar = tqdm(total=num_frames, desc="Generating side-by-side video")
    
    for idx in range(num_frames):
        orig_path = os.path.join(original_frames_dir, orig_files[idx])
        pred_path = os.path.join(predicted_bg_dir, pred_files[idx])
        
        original = cv2.imread(orig_path)
        predicted_bg = cv2.imread(pred_path)
        
        if original is None or predicted_bg is None:
            print(f"Skipping frame {idx} due to read error.")
            continue

        # Optionally, resize images to a standard size (here we assume each half is 720x480)
        original = cv2.resize(original, (1920, 1080))
        predicted_bg = cv2.resize(predicted_bg, (1920, 1080))
        
        # Compute the absolute difference between the original and the predicted background
        diff = cv2.absdiff(original, predicted_bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Threshold difference to obtain a binary foreground mask (adjust threshold as needed)
        _, fg_mask = cv2.threshold(diff_gray, 90, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), dtype=np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((float(x), float(y), float(x+w), float(y+h)))
                # Draw bounding box on the original frame (green rectangle)
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save bounding boxes for this frame for XML annotation
        annotations[idx] = bboxes
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        # Create side-by-side frame: left predicted background, right original with boxes
        side_by_side = np.hstack((fg_mask_bgr, original))
        side_by_side = cv2.resize(side_by_side, out_size)
        out_writer.write(side_by_side)
        pbar.update(1)
    
    pbar.close()
    out_writer.release()

    # Generate XML annotation file
    root_elem = ET.Element("annotations")
    for f_idx, boxes in annotations.items():
        frame_elem = ET.SubElement(root_elem, "frame", number=str(f_idx))
        for (xtl, ytl, xbr, ybr) in boxes:
            box_attrib = {
                "frame": str(f_idx),
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}",
                "outside": "0",
                "occluded": "0",
                "keyframe": "1"
            }
            ET.SubElement(frame_elem, "box", box_attrib)
    tree = ET.ElementTree(root_elem)
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    print(f"Side-by-side video saved to '{output_video_path}'")
    print(f"XML annotations saved to '{output_xml_path}'")

def predict_BMN(bmn,data_loader,exp_path):
    # create output directories
    print("saving results to:", exp_path)
    bg_est_path = os.path.join(exp_path, "background_estimation")
    pathlib.Path(bg_est_path).mkdir(parents=True, exist_ok=True)
    utils.save_image(os.path.join(exp_path, "panoramic_robust_mean.png"), bmn.moments[0])

    i = 0
    bmn.eval()
    for images in tqdm(data_loader, desc="predict background",total=len(data_loader)):
        images = images.to(device, dtype=torch.float32)
        background,_ = bmn.predict(images)
        for bg in background:
            name = f"frame{i:06}.png"
            save_path = os.path.join(bg_est_path, name)
            utils.save_image(save_path, bg)
            i = i +1 
    print("----------------------background estimation done----------------------")
    return bg_est_path

def main(args,**kwargs):
    ## Params and Settings 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    ## Data
    train_transforms = transforms.Compose([
        transforms.Resize(size=args.mask_shape),
        transforms.ToTensor()])
    root = os.path.join(args.parent_dir, args.dir, "frames")
    print(f"working on dataset in:{root}")
    predict_loader = DataLoader(
        ImageDataset(root=root,
                    transform=train_transforms),
        batch_size=1,#args.AE_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    # Model 
    checkpoint_path = os.path.join(args.BMN_ckpt_dir, args.BMN_ckpt)
    checkpoint = torch.load(checkpoint_path)
    # STN
    if "STN.homography_theta_regressor.0.weight" in checkpoint['state_dict'].keys():
        args.homography = True
        args.TG = "Homo"
        stn = STN_Homo.STN_Homo(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                    use_homography=args.homography).to(device) 
    else:                
        stn = STN.STN_block(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                    use_homography=args.homography).to(device) 
    # AE  
    in_chan = args.channels*(args.moments_num + 1)
    ae = AutoEncoder(C=args.C,
                        M=args.M,
                        in_chan=in_chan,
                        out_chan=args.channels,
                        input_shape=args.mask_shape,
                        code_size=args.code_size).to(device)
    # BMN using STN and AE 
    bmn = BMN(args.moments_num, stn, ae, use_theta_embedding=args.theta_embedding,
              cond_decoder=args.cond_decoder).to(device)
    bmn.load_state_dict(checkpoint['state_dict'])
    print("BMN was loaded from: ", checkpoint_path)
    bmn.init_moments(predict_loader,args.trim_percentage)

    dataset_results_dir = os.path.join(args.Results_dir,args.dir)
    exp = args.log_name
    utils.safe_mkdir(dataset_results_dir)
    exp_path = os.path.join(dataset_results_dir, exp)
    bg_est_path = predict_BMN(bmn,predict_loader,exp_path)
    # ---- Now generate side-by-side video and XML annotations ----
    # Assume original frames are in 'root' and predicted backgrounds are in 'bg_est_path'
    side_by_side_video_path = os.path.join(exp_path, "side_by_side.mp4")
    xml_output_path = os.path.join(exp_path, "annotations.xml")
    # You can adjust min_area as needed (here default is 500)
    generate_side_by_side_and_XML(original_frames_dir=root,
                                  predicted_bg_dir=bg_est_path,
                                  output_video_path=side_by_side_video_path,
                                  output_xml_path=xml_output_path,
                                  min_area=500,
                                  out_size=(1440, 480))
    gt_path = os.path.join(args.parent_dir,args.dir,"GT")    
    video_path = os.path.join(args.parent_dir, args.dir, "frames")
    mse_path = os.path.join(exp_path, "MSE")
    
    calc_metric_and_MSE(video_path=video_path, bg_path=bg_est_path,
                        gt_path=gt_path, mse_path=mse_path, args=args, method=args.method,
                        overwrite=True)

if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)