"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset
import random
from torch.utils.data import Sampler

#Constants
DEFAULT_STRIDE = 2      # Sampling stride (if greater than 1, frames are skipped) / Effectively reduces FPS
DEFAULT_TRAINING_OVERLAP = 0.9   # Temporal overlap between sampled clips (for traiing and validation only)
DEFAULT_EVAL_OVERLAP = 0

class CyclicChunkSampler(Sampler):
    def __init__(self, total_size, chunk_size):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self._global_indices = list(range(total_size))
        random.shuffle(self._global_indices)
        self._position = 0

    def __iter__(self):
        # If we reached the end, reshuffle and reset
        if self._position + self.chunk_size > self.total_size:
            random.shuffle(self._global_indices)
            self._position = 0

        chosen = self._global_indices[self._position : self._position + self.chunk_size]
        self._position += self.chunk_size

        return iter(chosen)
    
    def __len__(self):
        return self.chunk_size

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    print(args)
    epoch_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if "stride" in args else DEFAULT_STRIDE
    overlap = args.overlap if "overlap" in args else DEFAULT_TRAINING_OVERLAP

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task,
        'soft_action_dilation': args.soft_action_dilation,
    }

    print(dataset_kwargs)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, **dataset_kwargs)
    
    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, **dataset_kwargs)
      
    train_sampler = CyclicChunkSampler(len(train_data), epoch_len)
    val_sampler = CyclicChunkSampler(len(val_data), epoch_len//4)

    

    dataset_kwargs['overlap'] = DEFAULT_EVAL_OVERLAP

    eval_val_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, **dataset_kwargs)
    
    eval_test_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.clip_len, **dataset_kwargs)
    
    print()
    train_data.print_info()
    val_data.print_info()
    eval_test_data.print_info()

    print()
    print(f"Classes: {classes}")
    print()
    print(f'Train : Epoch Size - Total Size: {epoch_len} - {len(train_data)}')
    print()
    print(f'Val : Epoch Size - Total Size: {epoch_len//4}  - {len(val_data)}')
    print()
        
    return classes, train_data, train_sampler, val_data, val_sampler, eval_val_data, eval_test_data