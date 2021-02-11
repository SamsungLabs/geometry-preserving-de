import argparse
import os
import torch
from torch.utils.data import DataLoader
import tqdm

from datasets.validation_datasets import NYUDataset, TUMDataset, SintelDataset, ETH3DDataset
from datasets.diw_dataset_eval import DIW
import models
import metrics.logdisp_metrics as logdisp_metrics
import metrics.disp_metrics as disp_metrics
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, nargs='+', choices=list(DS_ZOO.keys()), default=list(DS_ZOO.keys()),
                        help='Datasets list to evaluate model')
    parser.add_argument('--model', type=str, choices=MODEL_ZOO, help='Model architecture to use')
    parser.add_argument('--checkpoint-dir', type=str, default='weights/', help='Folder with model checkpoints')
    parser.add_argument('--device', default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    return args


def get_dataset(ds_name):
    if ds_name == 'nyu':
        return NYUDataset(path=NYU_PATH, strategy='smallest_equal')
    elif ds_name == 'tum':
        return TUMDataset(path=TUM_PATH, strategy='smallest_equal')
    elif ds_name == 'sintel':
        return SintelDataset(path=SINTEL_PATH, strategy='smallest_equal')
    elif ds_name == 'eth3d':
        return ETH3DDataset(path=ETH3D_PATH, strategy='smallest_equal')
    elif ds_name == 'diw':
        return DIW(path=DIW_PATH)


@torch.no_grad()
def evaluate_dataset(ds_name, model, args):
    ds = get_dataset(ds_name)
    metric_name = DS_ZOO[ds_name]
    ds_loader = DataLoader(ds,
                           batch_size=16 if ds_name in ['nyu', 'tum', 'sintel'] else 1,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)
    uts_error, utss_error = 0.0, 0.0
    n = 0

    for batch in tqdm.tqdm(ds_loader, total=len(ds_loader), leave=False):
        for key in ['image', 'depth', 'mask']:
            batch[key] = batch[key].to(device=args.device)

        pred = model(batch['image'])
        val, norm = getattr(logdisp_metrics, metric_name)(pred, batch)
        uts_error += val.item()
        n += norm
        utss_error += getattr(disp_metrics, 'utss_' + metric_name)(pred.exp(), batch)[0]

    print('---- {}/{} -----'.format(args.model, ds_name))
    print('UTS-{} :  {:.2f}%'.format(metric_name, (uts_error / n) * 100.0))
    print('UTSS-{}:  {:.2f}%'.format(metric_name, (utss_error / n) * 100.0))


def main(args):
    assert args.model in MODEL_ZOO
    model = getattr(models, args.model)()
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model + '.pth'), map_location=args.device))
    model = model.to(device=args.device)
    model.eval()
    print('Using "{}" model'.format(args.model))

    for ds in args.ds:
        evaluate_dataset(ds, model, args)


if __name__ == '__main__':
    main(parse_args())
