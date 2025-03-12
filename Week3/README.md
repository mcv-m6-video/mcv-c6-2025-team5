# mcv-c6-2025-team5 Week 3

## Optical flow

### Off the shelf methods

- You can find an exploration of the ground truth pair and the metrics and results for the methods in the following [notebook](./dataset_flow/tester.ipynb)

### Executing them

#### Pyflow

#### RAFT

#### FlowFormer++

- Clone the following [repository](https://github.com/XiaoyuShi97/FlowFormerPlusPlus/blob/main/evaluate_FlowFormer_tile.py)
- Follow the installation instructions.
- Substitute the config file submission.py from the one on this repository.
- Have the required pair or sequence of images in a directory, use the vid2frame.py script if needed.
- Execute inside the repository the inf_pair.py script, use save_flow to save the original flow in npy format:

```
python inf_pair.py --seq_dir <SEQ_DIR>  --start_idx 0 --end_idx N --viz_root_dir <OUTPUT_DIR> [--save_flow]
```

#### NeuFlow2

- Clone the following [repository](https://github.com/neufieldrobotics/NeuFlow_v2)
- Follow the installation instructions.
- Have the required pair or sequence of images in a directory, use the vid2frame.py script if needed.
- Execute the infer_hf.py script, use save_flow to save the original flow in npy format:

```
python infer_hf.py --path <SEQ_DIR> [--save_flow]
```

## Object tracking with optical flow

Developed in the following colab: https://colab.research.google.com/drive/1GBvtkew2Ys5dOCm27enf6EHjuVNvFtoX?usp=sharing
