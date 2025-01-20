import numpy as np
import random,os,json
from lifelines.utils import concordance_index
import pandas
import torch
from sklearn.metrics import roc_curve, auc

def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def softmax(x):
    non_zeros=x[x!=0]
    e_x = np.exp(non_zeros)
    result=e_x / e_x.sum(axis=0)
    x[non_zeros.index]=result
    # x=result.fillna(0)
    return x


def cox_partial_likelihood_loss(risk_pred, time, event):

    order = torch.argsort(time, descending=True)
    sorted_risk_pred = risk_pred[order]
    sorted_event = event[order]

    log_cumulative_hazard = torch.logcumsumexp(sorted_risk_pred, dim=0)

    observed_risk = sorted_risk_pred[sorted_event == 1].sum()
    observed_hazard = log_cumulative_hazard[sorted_event == 1].sum()

    loss = - (observed_risk - observed_hazard)

    return loss




def calc_CI(df_pred, surv_type="OS", feature="risk"):

    time = df_pred[f"{surv_type}.time"]
    event = df_pred[surv_type]
    pred_scores = df_pred[feature]

    valid_indices=(~time.isna())&(~event.isna())&(~pred_scores.isna())
    time = time[valid_indices]
    event = event[valid_indices]
    pred_scores = pred_scores[valid_indices]
    ci = concordance_index(time, pred_scores, event)
    return ci

def roc_auc(df_pred, feature="risk"):

    y_true = df_pred["Responder"]
    y_score = df_pred[feature]
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def norm_weight(x,k=1):
    non_zeros=x[x!=0]*k
    e_x = np.exp(non_zeros)
    result=e_x / e_x.sum(axis=0)
    x[non_zeros.index]=result
    # x=result.fillna(0)
    return x