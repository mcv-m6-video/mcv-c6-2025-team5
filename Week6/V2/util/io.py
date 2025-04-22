import json
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
from matplotlib.colors import ListedColormap
import torch


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines

def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)

def save_labeled_images(images, labels, label_map, out_dir, preds=None):
    """
    Saves a list of torch images to disk with class names as titles.

    Args:
        images (List[Tensor]): List of torch tensors (C x H x W).
        labels (List[int]): List of integer labels.
        label_map (Dict[int, str]): Mapping from integer label to string label.
        out_dir (str): Folder to save the images in.
    """
    os.makedirs(out_dir, exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        class_name = label_map.get(label, "")
        img_np = TF.to_pil_image(torch.tensor(img)).convert("RGB")

        # Create plot
        plt.figure()
        plt.imshow(img_np)
        plt.title(class_name)
        plt.axis('off')
        plt.tight_layout()
        
        # Save image
        save_path = os.path.join(out_dir, f"image_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def save_images(images, out_dir):
    """
    Saves a list of torch images to disk with class names as titles.

    Args:
        images (List[Tensor]): List of torch tensors (C x H x W).
        labels (List[int]): List of integer labels.
        label_map (Dict[int, str]): Mapping from integer label to string label.
        out_dir (str): Folder to save the images in.
    """
    os.makedirs(out_dir, exist_ok=True)

    for idx, img in enumerate(images):
        # img_np = TF.to_pil_image(img*255).convert("RGB")

        # Create plot
        plt.figure()
        plt.imshow(img.mean(axis=0), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        
        # Save image
        save_path = os.path.join(out_dir, f"image_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def plot_3d_bar_targets(targets, output_path="targets_3d.png", cmap_name="tab10"):
    """
    Plots a 3D bar chart for a 2D array of shape (num_frames, num_classes),
    but swaps axes so the x-axis is 'class index' and the y-axis is 'frame index'.

    Args:
        targets (np.ndarray): 2D array of shape (T, C),
                              where T = number of frames, C = number of classes.
        output_path (str): Filename to save the figure.
    """
    if targets.ndim != 2:
        raise ValueError("targets must be 2D, shape (num_frames, num_classes).")

    

    T, C = targets.shape  # T=frames, C=classes

    # Create a meshgrid such that:
    #   - 'xpos' goes from 0..C-1 (class indices)
    #   - 'ypos' goes from 0..T-1 (frame indices)
    x_positions = np.arange(C)
    y_positions = np.arange(T)
    
    # 'indexing="xy"' => xpos.shape == (T, C), ypos.shape == (T, C),
    #   xpos[t, c] = c, ypos[t, c] = t
    xpos, ypos = np.meshgrid(x_positions, y_positions, indexing='xy')

    # Flatten for use in bar3D
    xpos_flat = xpos.ravel()
    ypos_flat = ypos.ravel()

    # All bars start at z=0
    zpos = np.zeros_like(xpos_flat)

    # Each bar is 0.6 in x,y directions
    dx = dy = 0.6

    # Heights = values from targets[t, c], matched with (ypos[t, c], xpos[t, c])
    dz = targets[ypos, xpos].ravel()

    cmap = plt.get_cmap(cmap_name)
    colors = []
    # Distinguish between ListedColormap vs. continuous colormap
    if isinstance(cmap, ListedColormap):
        # Discrete map: cycle through colors
        n_colors = len(cmap.colors)
        for i in range(len(xpos_flat)):
            class_idx = int(xpos_flat[i])
            color = cmap.colors[class_idx % n_colors]
            colors.append(color)
    else:
        # Continuous colormap: sample at equal intervals
        # e.g. class_idx / (C-1) if C>1, else 0.5
        for i in range(len(xpos_flat)):
            class_idx = int(xpos_flat[i])
            if C > 1:
                fraction = class_idx / (C - 1)
            else:
                fraction = 0.5
            color = cmap(fraction)
            colors.append(color)

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 3, 0.15))

    # Plot bars
    ax.bar3d(xpos_flat, ypos_flat, zpos, dx, dy, dz, color=colors, shade=True)

    # Label axes
    ax.set_xlabel("Class index")
    ax.set_ylabel("Frame index")

    ax.set_zticks([])

    # Optional: Use tight layout for better spacing
    plt.tight_layout()

    # Save and close
    plt.savefig(output_path)
    plt.close(fig)
