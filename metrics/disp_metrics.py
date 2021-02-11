import numpy as np
import torch
import torch.nn.functional as F


def align_mse(gt_elems, pred_elems):
    gt_sum = gt_elems.sum()
    pred_sum = pred_elems.sum()
    n_elems = torch.tensor(pred_elems.numel()).float().to(pred_elems.device)

    row1 = torch.stack([pred_elems.pow(2).sum(), pred_sum]).view(1, 2)
    row2 = torch.stack([pred_sum, n_elems]).view(1, 2)
    mat = torch.cat([row1, row2]).inverse()
    vec = torch.stack([(gt_elems * pred_elems).sum(), gt_sum]).view(2, 1)
    params = mat.mm(vec)

    return pred_elems * params[0, 0] + params[1, 0]


def depth_cap(one_pred, cap_value=None):
    pred_depth = 1.0 / torch.clamp(one_pred, 1.0 / cap_value, np.inf) if cap_value is not None else 1.0 / one_pred
    return pred_depth


def rel_core(one_pred, one_targ):
    return ((one_pred - one_targ).abs() / one_targ).sum()


def delta_core(one_pred, one_targ):
    return torch.max(one_pred / one_targ, one_targ / one_pred)


def d1_core(one_pred, one_targ):
    return (delta_core(one_pred, one_targ) > 1.25).float().sum()


def utss_metric(pred, input, f):
    res = torch.tensor(0.0)
    n = 1.0e-8

    for i in range(len(pred)):
        one_mask = input["mask"][i]
        one_depth = input["depth"][i][one_mask]
        one_pred = F.interpolate(pred[i].unsqueeze(0), size=input['depth'][i].shape[1:], mode='bilinear')[0][one_mask]

        if one_depth.numel() > 10:
            pred_aligned = align_mse(one_depth, one_pred)
            pred_aligned = depth_cap(pred_aligned, cap_value=input['depth_cap'][0, 0].item() if 'depth_cap' in input else None)
            res = res + f(pred_aligned, 1.0 / one_depth) / one_pred.numel()
            n = n + 1

    return res, n


def utss_d1(pred, input):
    return utss_metric(pred, input, d1_core)


def utss_rel(pred, input):
    return utss_metric(pred, input, rel_core)


def utss_whdr(pred, input):
    res = torch.tensor(0.0)
    n = 1.0e-8

    for i in range(len(pred)):
        one_pred = F.interpolate(pred[i].unsqueeze(0), size=input['depth'][i].shape[1:], mode='bilinear')[0,0]
        ya, xa, yb, xb, rel = input['ordinal'][i][0]
        res += ((one_pred[yb, xb] > one_pred[ya, xa]).item() != (rel>0).item())
        n += 1

    return res, n