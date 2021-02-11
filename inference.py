import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

import models
from datasets.validation_datasets import FolderDataset
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=str, required=True, help='Source folder with images')
    parser.add_argument('--out-dir', type=str, required=True, help='Folder to put output predictions')
    parser.add_argument('--vis-dir', type=str, help='Folder to put visualisations')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualisations (only predictions)')
    parser.add_argument('--model', type=str, choices=MODEL_ZOO, help='Model architecture to use')
    parser.add_argument('--checkpoint-dir', type=str, default='weights/', help='Folder with model checkpoints')
    parser.add_argument('--domain', type=str, choices=['depth', 'log-depth', 'disparity', 'log-disparity'],
                        default='log-disparity', help='Which domain to use to store predictions')
    parser.add_argument('--device', default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    return args


def convert_predictions(pred, domain):
    if domain == 'depth':
        return (-pred).exp()
    elif domain == 'log-depth':
        return -pred
    elif domain == 'disparity':
        return pred.exp()
    else:
        return pred


def get_path(path, old_dir, new_dir, postfix):
    pred_path = os.path.relpath(path, old_dir)
    pred_path = os.path.join(new_dir, os.path.dirname(pred_path),
                             os.path.splitext(os.path.basename(pred_path))[0] + postfix)
    return pred_path


def save_vis(pred, out_path):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    vmin, vmax = np.quantile(pred, [0.01, 0.99])
    vis = ((torch.clamp(pred, vmin, vmax) - vmin) * 255 / (vmax - vmin)).numpy().astype(np.uint8)
    cv2.imwrite(out_path, cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO))


def save_pred(pred, out_path):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    torch.save(pred, out_path)


@torch.no_grad()
def main(args):
    ds = FolderDataset(args.src_dir)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print('Found {} images in source directory: {}'.format(len(ds), args.src_dir))

    assert args.model in MODEL_ZOO
    model = getattr(models, args.model)()
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model + '.pth'), map_location=args.device))
    model = model.to(device=args.device)
    model.eval()
    print('Using "{}" model (predicting in {} domain)'.format(args.model, args.domain))

    if (not args.no_vis) and (args.vis_dir is None):
        args.vis_dir = args.out_dir

    for batch in tqdm.tqdm(ds_loader, total=len(ds_loader)):
        pred = model(batch['image'].to(device=args.device))
        pred = convert_predictions(pred, args.domain).cpu()

        for i in range(len(pred)):
            save_pred(pred[i, 0], get_path(batch['path'][i], args.src_dir, args.out_dir, '.pth'))
            if not args.no_vis:
                save_vis(pred[i, 0], get_path(batch['path'][i], args.src_dir, args.vis_dir, '_vis.png'))


if __name__ == '__main__':
    main(parse_args())
