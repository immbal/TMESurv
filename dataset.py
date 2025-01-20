import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import pandas as pd

import config
from preprocess import mask1,mask2,mask3
from config import  BATCH_SIZE,DATASET,SURVIVAL_TYPE,TEST_DATASET

#############################
## load mask info.
#############################

fc1_row,fc1_col=mask1.index,mask1.columns
fc2_row,fc2_col=mask2.index,mask2.columns
fc3_row,fc3_col=mask3.index,mask3.columns


mask1=np.array(mask1)
mask2=np.array(mask2)
mask3=np.array(mask3)



class Ds(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __getitem__(self, index):
        return torch.tensor(self.X[index],dtype=torch.float),\
            torch.tensor(self.y[index],dtype=torch.float)

    def __len__(self):
        return len(self.X)
##load expression data

def generate_train_test(rawfile,_seed=42,_ratio=0.7):
    # global y, dataset, train_dataset, test_dataset
    df = pd.read_csv(rawfile, sep=",", header=0, index_col=0)
    diff_genes = set(fc1_row).difference(df.columns)
    # datasets=df["Dataset"].unique().tolist()
    # training_indices = df.index[df['Dataset'] != TEST_DATASET].tolist()
    # test_indices = df.index[df['Dataset'] == TEST_DATASET].tolist()
    ############################################
    ### Handling and Filtering NaN Data
    ############################################
    df=df[~(df[SURVIVAL_TYPE].isna() | df["{0}.time".format(SURVIVAL_TYPE)].isna())]

    ### feature alignment
    X = df[fc1_row]
    y = df[["{0}.time".format(SURVIVAL_TYPE), SURVIVAL_TYPE]]
    X = np.array(X)
    y = np.array(y)
    dataset = Ds(X, y)
    train_size = int(_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator1 = torch.Generator().manual_seed(_seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)

    return train_dataset, test_dataset

# generate_train_test()

# test_mask = (df["Dataset"] == TEST_DATASET)
# X_train, X_test = X[~test_mask], X[test_mask]
# y_train, y_test = y[~test_mask], y[test_mask]
# train_dataset=Ds(np.array(X_train),np.array(y_train))
# test_dataset=Ds(np.array(X_test),np.array(y_test))

def scale_df(tcga,columns_to_normalize=fc1_row):
    df=pd.read_csv("data/tcga_{0}.csv".format(tcga.lower()), sep=",", header=0, index_col=0)
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df


def generate_pan_train_test(test_type="BLCA"):
    train_types=[ x for x in config.TCGA_STUDYS if x !=test_type]
    df_train=[ scale_df(x)  for x in train_types ]
    df_train=pd.concat(df_train)
    X = df_train[fc1_row]
    y = df_train[["{0}.time".format(SURVIVAL_TYPE), SURVIVAL_TYPE]]
    X = np.array(X)
    y = np.array(y)

    ## shuffle data
    shuffle_indices = np.random.permutation(X.shape[0])
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    train_dataset = Ds(X,y)


    df_test=scale_df(test_type)
    X = df_test[fc1_row]
    y = df_test[["{0}.time".format(SURVIVAL_TYPE), SURVIVAL_TYPE]]
    X = np.array(X)
    y = np.array(y)
    test_dataset =  Ds(X,y)

    return train_dataset, test_dataset