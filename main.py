
import sys,os
from benchmark import train_TMENET, train_FullNetwork, train_DeepSurvNetwork, train_CoxPH, train_RSF
from datetime import datetime
import config
from dataset import generate_train_test

SEED = int(sys.argv[1])
DATASET_FILE=  sys.argv[2]

train_dataset, test_dataset= generate_train_test(DATASET_FILE, _seed=SEED, _ratio=config.TRAIN_TEST_RATIO)

### traning for model tmenet
TMENET_CI,TMENET_HR,TMENET_AUC=train_TMENET( train_dataset,test_dataset)
### traning for full connected network
FULL_CI,FULL_HR,FULL_AUC=train_FullNetwork(train_dataset,test_dataset)
### traning for dropout network
DROP_CI,DROP_HR,DROP_AUC=train_DeepSurvNetwork(train_dataset, test_dataset)
### traning for COX
COX_CI,COX_HR,COX_AUC=train_CoxPH(train_dataset,test_dataset)
### traning for RSF
RSF_CI,RSF_HR,RSF_AUC=train_RSF(train_dataset,test_dataset)
logtime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

filename=os.path.splitext(os.path.basename(DATASET_FILE))[0]
with open("data/{0}/log_{1}.txt".format(config.SURVIVAL_TYPE,filename), "a", encoding="utf-8") as f:
    f.write(f"{SEED}\t{1-TMENET_CI}\t{1-FULL_CI}\t{1-DROP_CI}\t{1-COX_CI}\t{RSF_CI}\t{logtime}\n" )

with open("data/{0}/log_{1}_HR.txt".format(config.SURVIVAL_TYPE,filename), "a", encoding="utf-8") as f:
    f.write(f"{SEED}\t{TMENET_HR}\t{FULL_HR}\t{DROP_HR}\t{COX_HR}\t{RSF_HR}\t{logtime}\n" )

with open("data/{0}/log_{1}_AUC.txt".format(config.SURVIVAL_TYPE,filename), "a", encoding="utf-8") as f:
    f.write(f"{SEED}\t{TMENET_AUC[365]}\t{FULL_AUC[365]}\t{DROP_AUC[365]}\t{COX_AUC[365]}\t{RSF_AUC[365]}\t{logtime}\n" )

with open("data/{0}/log_{1}_1825_AUC.txt".format(config.SURVIVAL_TYPE,filename), "a", encoding="utf-8") as f:
    f.write(f"{SEED}\t{TMENET_AUC[1825]}\t{FULL_AUC[1825]}\t{DROP_AUC[1825]}\t{COX_AUC[1825]}\t{RSF_AUC[1825]}\t{logtime}\n" )