import pandas as pd
from config import EMBEDDING_LEN

file=r"data\chatGPT_markers.csv"

df_meta_info=pd.read_csv(file)
df_meta_info["component"]=df_meta_info.apply(lambda data:"{0}_{1}".format(data["cell"],data["component"]),axis=1)
###############
# mask1
###############
df_mask1=df_meta_info[["gene","component"]].drop_duplicates( )
df_mask1=pd.pivot_table(df_mask1,index='gene',columns='component',aggfunc=len,fill_value=0)
###############
# mask2
###############
df_mask2=df_meta_info[["component","cell"]].drop_duplicates( )
df_mask2=pd.pivot_table(df_mask2,index='component',columns='cell',aggfunc=len,fill_value=0)
df_mask2=df_mask2.reindex(df_mask1.columns)
df_mask2=df_mask2[['CD8+ T cells', 'M1 macrophages','NK cells','Th1 cells',
       'Myeloid-Derived Suppressor Cells (MDSC)',  'M2 macrophages','Cancer-Associated Fibroblasts (CAFs)',
       'Regulatory T cells (Tregs)']]
###############
# mask3
###############

df_cell_role=df_meta_info[["cell","role"]].drop_duplicates( )
df_cell_role=pd.pivot_table(df_cell_role,index='cell',columns='role',aggfunc=len,fill_value=0)
df_cell_role=df_cell_role.reindex(df_mask2.columns)
df_mask3=pd.concat([df_cell_role["Immunostimulatory"]]*EMBEDDING_LEN +[df_cell_role["Immunosuppressive"]]*EMBEDDING_LEN ,axis=1)
df_mask3.columns=["embedding_{0:02d}".format(x) for x in range(1,2*EMBEDDING_LEN+1) ]

df_mask1.to_csv("../data/mask1.csv" )
df_mask2.to_csv("../data/mask2.csv" )
df_mask3.to_csv("../data/mask3.csv" )