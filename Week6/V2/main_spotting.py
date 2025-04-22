#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

#Local imports
from util.io import load_json, store_json, save_labeled_images, plot_3d_bar_targets
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model
from tqdm import tqdm


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.stride = config['stride']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    args.loss = config['loss'] if 'loss' in config else 'default'
    args.soft_action_dilation = config['soft_action_dilation'] if 'soft_action_dilation' in config else 0
    args.background_weight = config['background_weight'] if 'background_weight' in config else 1
    args.model_impl = config['model_impl'] if 'model_impl' in config else 'baseline'

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def compute_class_distribution(dataset, classes, label_key='target', max_iterations=2000):
    """
    Computes class counts and frequency over the entire dataset.

    Args:
        dataset (Dataset or DataLoader): PyTorch dataset or dataloader that yields batches with labels.
        num_classes (int): Number of classes (excluding background if not included).
        label_key (str): Key used to access labels in dataset item dicts.
        include_background (bool): Whether to include background class (assumed index 0).

    Returns:
        counts (Tensor): Raw class counts, shape (C,)
        freqs (Tensor): Normalized class frequencies, shape (C,)
    """
    print("Computing class distribution...")
    total_classes = len(classes.keys())
    counts = np.ones(total_classes+1)

    for i in tqdm(range(max_iterations)):
        sample = dataset[i]
        targets = sample[label_key]
        counts += np.sum(targets, axis=0)

    total = counts.sum()
    freqs = (counts + 1) / (total + 1)
    inv_freq = (total + 1) / (counts + 1)
    inv_freq /= inv_freq.mean()
    bf_counts = np.array([counts[0], counts[1:].sum()])
    bf_freq = bf_counts / bf_counts.sum()
    equalized_frequency = np.zeros(total_classes+1)
    equalized_frequency[0] = bf_freq[0]
    equalized_frequency[1:] = bf_freq[1] / total_classes
    inv_equalized_freq = (1 / bf_freq) / np.max((1 / bf_freq))

    table = [["Background", f"{freqs[0]*100:.3f}", f"{equalized_frequency[0]*100:.3f}", f"{inv_equalized_freq[0]:.4f}"]]
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{freqs[i+1]*100:.3f}", f"{equalized_frequency[i+1]*100:.3f}", f"{inv_equalized_freq[1]:.4f}"])
    headers = ["Class", "Frequency", "Equalized Inverse Frequency", "Equalized Inverse Weights"]
    print(tabulate(table, headers, tablefmt="grid"))

def run_evaluation(model, test_data, classes, workers=1):
    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_data, nms_window = 5, workers=workers)

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report average results in table
    avg_table = [["Mean", f"{map_score*100:.2f}"]]
    headers = ["", "Average Precision"]

    print(tabulate(avg_table, headers, tablefmt="grid"))
    return map_score

def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    torch.set_printoptions(sci_mode=False)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, train_sampler, val_data, val_sampler, eval_val_data, eval_test_data = get_datasets(args)
    # print(classes)
    # compute_class_distribution(train_data, classes)
    # print(train_data._frame_paths)
    # print("Train Dataset Length: ", len(train_data))
    sample_frames = np.concatenate([val_data[0]['frame'], val_data[1]['frame']])
    sample_labels = np.concatenate([val_data[0]['label'], val_data[1]['label']])
    # plot_3d_bar_targets(sample['target'], 
    #                     "/ghome/c5mcv05/Eudald/CVMasterActionSpotting/_output/figures/targets.png",
    #                     cmap_name='tab20')
    # print(sample['label'])
    # save_labeled_images(
    #     sample_frames, 
    #     sample_labels, 
    #     {v: k for k, v in classes.items()}, 
    #     "/ghome/c5mcv05/Eudald/CVMasterActionSpotting/_output/figures")

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, 
        shuffle=False, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True, 
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_data, 
        shuffle=False, 
        batch_size=args.batch_size,
        sampler=val_sampler,
        pin_memory=True, 
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('-inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)

            maps = run_evaluation(model, eval_val_data, classes, workers=args.num_workers)

            better = False
            if maps > best_criterion:
                best_criterion = maps
                better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} Map: {:0.5f}'.format(
                epoch, train_loss, val_loss, maps))
            if better:
                print('New best epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss, 'map': maps,
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    run_evaluation(model, eval_test_data, classes, workers=args.num_workers)
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())