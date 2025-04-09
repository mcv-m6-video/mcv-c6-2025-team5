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
import wandb
import time

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model


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
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    args.num_heads_transformer = config['num_heads_transformer']
    args.num_layers_transformer = config['num_layers_transformer']
    args.num_global_tokens = config["num_global_tokens"]
    args.attention_window_size  = config["attention_window_size"]
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


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)


    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"model_{args.model}_run_{run_id}"
    wandb.init(project="action-spotting-local-transformer", name=run_name, config=vars(args),dir=args.save_dir)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data, val_extra_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
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
        best_map10 = -1.0  # track best mAP
        epoch = 0
        patience = 4
        bad_epochs = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_map, val_ap_scores = evaluate(model, val_extra_data, nms_window=5)
            val_loss = model.epoch(val_loader)

            ap10_scores = []
            for i, class_name in enumerate(classes.keys()):
                if class_name not in ['FREE KICK', 'GOAL']:
                    ap10_scores.append(val_ap_scores[i])
            val_map10 = np.mean(ap10_scores)


            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} Val mAP: {:0.5f} Val mAP@10: {:0.5f}'.format(
                epoch, train_loss, val_loss, val_map, val_map10))

            better = val_map10 > best_map10
            if better:
                best_map10 = val_map10
                bad_epochs = 0
                print('New best mAP epoch!')
            else:
                bad_epochs += 1
                print(f'No improvement in mAP@10. ({bad_epochs}/{patience} bad epochs)')

            losses.append({
                'epoch': epoch,
                'train': train_loss,
                'val': val_loss,
                'val_map': val_map,
                'val_map10': val_map10
            })


            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_map": val_map,
                'val_map10': val_map10,
                "best_val_map10": best_map10,
                "learning_rate": current_lr
            }, step=epoch)

            for i, class_name in enumerate(classes.keys()):
                wandb.log({f"AP/{class_name}": val_ap_scores[i] * 100}, step=epoch)

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))
            
            # Early stopping
            if bad_epochs >= patience:
                print(f'Stopping early at epoch {epoch} due to no improvement in mAP@10 for {patience} consecutive epochs.')
                break
            


    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_data, nms_window = 5)

    # Report results per-class in table
    table = []
    table10 = []
    ap_list10 = []

    for i, class_name in enumerate(classes.keys()):
        ap_percent = ap_score[i] * 100
        table.append([class_name, f"{ap_percent:.2f}"])

        if class_name not in ['FREE KICK', 'GOAL']:
            table10.append([class_name, f"{ap_percent:.2f}"])
            ap_list10.append(ap_score[i])

        # Log each class AP to W&B
        wandb.log({f"Test AP/{class_name}": ap_percent})

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Full mAP
    avg_map = np.mean(ap_score) * 100
    avg_table = [["Average", f"{avg_map:.2f}"]]

    # mAP@10 (excluding "FREE KICK" and "GOAL")
    avg_map10 = np.mean(ap_list10) * 100
    avg_table10 = [["Average10", f"{avg_map10:.2f}"]]

    print(tabulate(avg_table, headers, tablefmt="grid"))
    print(tabulate(avg_table10, headers, tablefmt="grid"))

    # Log to W&B
    wandb.log({
        "test_map": map_score,                  # raw map (0-1)
        "test_map_percent": avg_map,            # in percent
        "test_map10_percent": avg_map10         # map@10 in percent
    })
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')
    wandb.finish()


if __name__ == '__main__':
    main(get_args())