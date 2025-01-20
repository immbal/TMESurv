import os
import sys

import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from torch.utils.data import DataLoader

import config
from dataset import fc1_row, fc1_col, fc2_row, fc2_col, fc3_row, fc3_col, mask1
from model import DeepSurvNetwork, TmeNet, FullNetwork
from utils import cox_partial_likelihood_loss, calc_CI, norm_weight
from lifelines import CoxPHFitter
import numpy as np
def set_seed(seed=42):
    torch.manual_seed(seed)

def train_network(model, train_dataset, test_dataset, epochs=100, lr=0.01, L1=22, calculate_extra_terms=False, keep_weight=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_index, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            NE = (target[:, 1] == 1).sum()
            N = target.shape[0]
            # Loss Calculation
            if calculate_extra_terms:
                out, cosine, sti, supp = outputs
                loss = cox_partial_likelihood_loss(out, target[:, 0], target[:, 1])/NE + torch.sum((-1 - cosine) ** 2) * L1/N
            else:
                out = outputs
                loss = cox_partial_likelihood_loss(out, target[:, 0], target[:, 1])/NE

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        # print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}")
    #save weight.
    if keep_weight:
        save_model_weight(model)

    # Evaluation
    return evaluate_model(model,test_dataset,calculate_extra_terms)


def evaluate_model(model,test_dataset,calculate_extra_terms=False):
    model.eval()
    test_dataset_x=torch.tensor(test_dataset.dataset.X[test_dataset.indices] ,dtype=torch.float)
    test_dataset_y=torch.tensor(test_dataset.dataset.y[test_dataset.indices] ,dtype=torch.float)

    if calculate_extra_terms:
        risk, cosine, sti, supp = model(test_dataset_x)
        risk = risk.detach().numpy().reshape(-1)
        sti = sti.detach().numpy().reshape(-1)
        supp = supp.detach().numpy().reshape(-1)
    else:
        risk = model(test_dataset_x)
        risk = risk.detach().numpy().reshape(-1)

    output_result = pd.DataFrame({"risk":risk,"{0}.time".format(config.SURVIVAL_TYPE):test_dataset_y[:,0],config.SURVIVAL_TYPE:test_dataset_y[:,1]})

    if calculate_extra_terms:
        output_result["sti"] = sti
        output_result["supp"] = supp
        ## append extra interpretable info.
        log_more_performance(output_result)

    cIndex = calc_CI(output_result, surv_type=config.SURVIVAL_TYPE, feature="risk")

    ## hr value
    hr = model_hazard_ratio(risk,test_dataset_y[:,0],test_dataset_y[:,1])

    auc=survival_auc(output_result["{0}.time".format(config.SURVIVAL_TYPE)],risk )

    return cIndex,hr,auc


def train_TMENET(train_dataset, test_dataset, epochs=100, lr=0.01, L1=33):
    model = TmeNet()
    return train_network(model, train_dataset, test_dataset, epochs, lr, L1, calculate_extra_terms=True, keep_weight=True)

def train_TMENET_Raw(train_dataset, test_dataset, epochs=100, lr=0.01, L1=0):
    model = TmeNet()
    return train_network(model, train_dataset, test_dataset, epochs, lr, L1, calculate_extra_terms=True)


def train_FullNetwork(train_dataset, test_dataset, epochs=100, lr=0.01, L1=33):
    model = FullNetwork()
    return train_network(model, train_dataset, test_dataset, epochs, lr, L1)


def train_DeepSurvNetwork(train_dataset, test_dataset, epochs=100, lr=0.01, L1=33):
    model = DeepSurvNetwork()
    return train_network(model, train_dataset, test_dataset, epochs, lr, L1)

def model_hazard_ratio(risk_pred,time,event):
    df=pd.DataFrame({"risk":risk_pred,"time":time,"event":event})
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    hr_value = cph.hazard_ratios_["risk"] # Extract the hazard ratio for 'risk'

    # Check for abnormal HR values
    if hr_value > 10 or hr_value < 0.1:  # Define threshold for abnormal HR
        hr_value = np.nan  # Set HR to NaN if it's outside the reasonable range
    return hr_value

def survival_auc(times, risk_pred):

    auc_results = {}
    for t in config.TIME_HORIZONS:
        binary_event = (times <= t).astype(int)  # Mark events occurring within the time horizon as 1, otherwise as 0
        # Check if binary_event has only one unique value
        if len(np.unique(binary_event)) == 1:
            auc = np.nan  # Set AUC to NaN if binary_event contains only one value
        else:
            auc = roc_auc_score(binary_event, risk_pred)
        auc_results[t] = auc
    return auc_results

def log_more_performance(output_result):

    cIndex_risk = calc_CI(output_result, surv_type=config.SURVIVAL_TYPE, feature="risk")
    cIndex_sti = calc_CI(output_result, surv_type=config.SURVIVAL_TYPE, feature="sti")
    cIndex_supp = calc_CI(output_result, surv_type=config.SURVIVAL_TYPE, feature="supp")

    ## hr value
    hr_risk = model_hazard_ratio(output_result["risk"], output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result[config.SURVIVAL_TYPE])
    hr_sti = model_hazard_ratio(output_result["sti"], output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result[config.SURVIVAL_TYPE])
    hr_supp = model_hazard_ratio(output_result["supp"], output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result[config.SURVIVAL_TYPE])

    auc_risk = survival_auc(output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result["risk"])
    auc_sti = survival_auc(output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result["sti"])
    auc_supp = survival_auc(output_result["{0}.time".format(config.SURVIVAL_TYPE)], output_result["supp"])

    name = os.path.basename(sys.argv[2]).split(".")[0]
    with open("data/interpretable/log_{0}.txt".format(name), "a", encoding="utf-8") as f:
        f.write(
            f"{1 - cIndex_risk}\t{ cIndex_sti}\t{1 - cIndex_supp}\t{hr_risk}\t{hr_sti}\t{hr_supp}\t{auc_risk[365]}\t{1-auc_sti[365]}\t{auc_supp[365]}\t{auc_risk[730]}\t{1-auc_sti[730]}\t{auc_supp[730]}\t{auc_risk[1825]}\t{1-auc_sti[1825]}\t{auc_supp[1825]}\n")

def save_model_weight(model):
    model.eval()
    name = os.path.basename(sys.argv[2]).split(".")[0]
    if( not os.path.exists("data/weight/{0}".format(name))):
        os.mkdir("data/weight/{0}".format(name))

    fc1 = pd.DataFrame(model.fc1.weight.detach().clone().numpy())
    fc1.columns = fc1_row
    fc1.index = fc1_col
    # fc1 = fc1.apply(norm_weight, axis=1)
    fc1.to_csv("data/weight/{0}/fc1_{1:03d}.csv".format(name,int(sys.argv[1])))

    fc2 = pd.DataFrame(model.fc2.weight.detach().clone().numpy())
    fc2.columns = fc2_row
    fc2.index = fc2_col
    # fc2 = fc2.apply(norm_weight, axis=1)
    fc2.to_csv("data/weight/{0}/fc2_{1:03d}.csv".format(name,int(sys.argv[1])))



def train_CoxPH( train_dataset, test_dataset):
    time=f"{config.SURVIVAL_TYPE}.time"
    event=config.SURVIVAL_TYPE
    columns = fc1_row.to_list() + [time, event]
    train_data= pd.DataFrame(np.hstack([train_dataset.dataset.X,train_dataset.dataset.y])[train_dataset.indices],columns=columns)
    test_data= pd.DataFrame(np.hstack([test_dataset.dataset.X,test_dataset.dataset.y] )[test_dataset.indices],columns=columns)
    cph = CoxPHFitter()
    cph.fit(train_data, duration_col=time, event_col=event)
    # cph.print_summary()

    predicted_risk = cph.predict_partial_hazard(test_data)
    cIndex = concordance_index(test_data[time], predicted_risk, test_data[event])
    hr=model_hazard_ratio(predicted_risk,test_data[time],test_data[event])
    auc=survival_auc(test_data[time],predicted_risk )
    return cIndex,hr,auc

def train_RSF( train_dataset, test_dataset):
    time=f"{config.SURVIVAL_TYPE}.time"
    event=config.SURVIVAL_TYPE
    columns = fc1_row.to_list() + [time,event]

    train_data= pd.DataFrame(np.hstack([train_dataset.dataset.X,train_dataset.dataset.y])[train_dataset.indices],columns=columns)
    test_data= pd.DataFrame(np.hstack([test_dataset.dataset.X,test_dataset.dataset.y] )[test_dataset.indices],columns=columns)

    X_train = train_data.drop(columns=[time,event])
    y_train = Surv.from_dataframe(event, time,  train_data[[time, event]])

    X_test = test_data.drop(columns=[time,event])
    y_test = Surv.from_dataframe(event, time, test_data[[time, event]])

    rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
    rsf.fit(X_train, y_train)

    y_pred = rsf.predict(X_test)
    cIndex = concordance_index_censored(y_test[event], y_test[time], y_pred)[0]
    hr=model_hazard_ratio(y_pred,y_test[time],y_test[event])
    auc=survival_auc(y_test[time],y_pred )
    return cIndex,hr,auc
