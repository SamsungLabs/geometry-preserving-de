import numpy as np
import torch
import torch.nn.functional as F


def log10_core(one_pred, one_targ):
    return (one_pred.log10() - one_targ.log10()).abs().sum()
    

def rel_core(one_pred, one_targ):
    return ((one_pred - one_targ).abs() / one_targ).sum()


def delta_core(one_pred, one_targ):
    return torch.max(one_pred / one_targ, one_targ / one_pred)


def d1_core(one_pred, one_targ):
    return (delta_core(one_pred, one_targ) > 1.25).float().sum()


def d2_core(one_pred, one_targ):
    return (delta_core(one_pred, one_targ) > 1.25 ** 2).float().sum()


def d3_core(one_pred, one_targ):
    return (delta_core(one_pred, one_targ) > 1.25 ** 3).float().sum()


def log_rel_metric(pred, input, f):
    res = torch.tensor(0.0)
    n = 1.0e-8
    
    for i in range(len(pred)):
        sample_type = input["type"][i]

        if sample_type == 0 or sample_type == 1:
            one_mask = input["mask"][i]
            one_depth = -input["depth"][i][one_mask].log()
            one_pred = F.interpolate(pred[i].unsqueeze(0), size=input['depth'][i].shape[1:], mode='bilinear')[0]
            one_pred = -one_pred[one_mask]

            if one_depth.numel() > 10:
                one_pred = one_pred + (one_depth - one_pred).median()
                res = res + f(one_pred.exp(), one_depth.exp()) / one_pred.numel()
                n = n + 1
    
    return res, n


def log10(pred, input):
    return log_rel_metric(pred, input, log10_core)


def d1(pred, input):
    return log_rel_metric(pred, input, d1_core)


def d2(pred, input):
    return log_rel_metric(pred, input, d2_core)


def d3(pred, input):
    return log_rel_metric(pred, input, d3_core)


def rel(pred, input):
    return log_rel_metric(pred, input, rel_core)


def whdr(pred, input):
    res = torch.tensor(0.0)
    n = 1.0e-8

    for i in range(len(pred)):
        one_pred = F.interpolate(pred[i].unsqueeze(0), size=input['depth'][i].shape[1:], mode='bilinear')[0,0]
        ya, xa, yb, xb, rel = input['ordinal'][i][0]
        res += ((one_pred[yb, xb] > one_pred[ya, xa]).item() != (rel>0).item())
        n += 1

    return res, n
