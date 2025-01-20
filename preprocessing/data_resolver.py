import os,re
from collections import Counter
import pandas as pd
from data_extraction import cells
### text analysis
def extract_gene_freq(lines,label="Secreted: ",cutoff=50):
    data = [line.replace(label, "").split(",") for line in lines if line.startswith(label)]
    all_genes = [gene.strip().upper() for sublist in data for gene in sublist]
    gene_counters = dict(Counter(all_genes))
    gene_counters={k:v for k,v in gene_counters.items() if v>=cutoff}
    return gene_counters



files=os.listdir(r"data\chatGPT")
maker_genes = []
for file in files:
    with open(r"D:\project\reference\data\chatGPT\{0}".format(file),"r",encoding="utf-8") as f:
        lines =[ line.strip( ) for line in f.readlines()]
        secreted= extract_gene_freq(lines,label="Secreted: ")
        maker_genes=maker_genes+[ {"gene":k,"freq":v,"component":"secreted","cell":os.path.splitext(file)[0].strip()} for k,v in secreted.items()]
        surface=extract_gene_freq(lines,label="Surface: ")
        maker_genes=maker_genes+[ {"gene":k,"freq":v,"component":"surface","cell":os.path.splitext(file)[0].strip()} for k,v in surface.items()]
        transfactor=extract_gene_freq(lines,label="Transcription factor: ")
        maker_genes=maker_genes+[ {"gene":k,"freq":v,"component":"transfactor","cell":os.path.splitext(file)[0].strip()} for k,v in transfactor.items()]

df_marker_genes = pd.DataFrame(maker_genes)
df_marker_genes["role"]=df_marker_genes.apply(lambda data: "Immunostimulatory" if data["cell"] in cells["Immunostimulatory"] else "Immunosuppressive",axis=1)

df_marker_genes.to_csv(r"data\chatGPT_markers.csv",index=False,encoding="utf-8")


