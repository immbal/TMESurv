#trainning settings.
import sys

TRAIN_TEST_RATIO =0.8

# SEED=42
BATCH_SIZE=64
EMBEDDING_LEN=8

DATASET= "data/tcga_skcm.csv"
SURVIVAL_TYPE="OS"
#'Gide_Cell_2019','HugoLo_IPRES_2016','Liu_NatMed_2019','Riaz_Nivolumab_2017','VanAllen_antiCTLA4_2015'
TEST_DATASET="Gide_Cell_2019"

TCGA_STUDYS=["BLCA", "COAD","HNSC","KIRC","LGG","LUAD","LUSC","OV", "SKCM","STAD" ]

TIME_HORIZONS=[ 365, 730, 1095, 1825]



