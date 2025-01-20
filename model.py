import torch
from torch import nn

from CustomLinear  import CustomizedLinear
from config import EMBEDDING_LEN
from  dataset import mask1,mask2,mask3

class TmeNet(nn.Module):
    def __init__(self,  out_size=1 ):
        super(TmeNet, self).__init__()
        self.fc1 = CustomizedLinear(mask=mask1)
        self.bn1=nn.BatchNorm1d(mask1.shape[1])
        self.fc2 = CustomizedLinear(mask=mask2)
        self.bn2=nn.BatchNorm1d(mask2.shape[1])
        self.fc3 = CustomizedLinear(mask=mask3)
        self.fc4 = nn.Linear(mask3.shape[1]  ,out_size)
        self.relu = nn.GELU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x= self.bn1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x= self.bn2(x)
        x = self.tanh(x)

        x = self.fc3(x)
        x = self.tanh(x)

        x_Immunostimulatory = x[:, :EMBEDDING_LEN]
        x_Immunosuppressive = x[:, EMBEDDING_LEN:]

        norm_Immunostimulatory = torch.norm(x_Immunostimulatory, dim=1)
        norm_Immunosuppressive = torch.norm(x_Immunosuppressive, dim=1)

        difference =  norm_Immunosuppressive -norm_Immunostimulatory
        # # x=x[:,:8]-x[:,8:]
        #
        cos_similarity = torch.cosine_similarity(x_Immunostimulatory, x_Immunosuppressive, dim=1)

        #torch.sum(x_Immunostimulatory * x_Immunosuppressive, dim=1)
        # x = self.fc4(x)
        return   difference,cos_similarity,norm_Immunostimulatory,norm_Immunosuppressive



class DeepSurvNetwork(nn.Module):
    def __init__(self,  out_size=1 ):
        super(DeepSurvNetwork, self).__init__()
        self.fc1 = nn.Linear(mask1.shape[0],mask1.shape[1])
        self.bn1=nn.BatchNorm1d(mask1.shape[1])
        self.fc2 = nn.Linear(mask2.shape[0],mask2.shape[1])
        self.bn2=nn.BatchNorm1d(mask2.shape[1])
        self.fc3 = nn.Linear(mask3.shape[0],mask3.shape[1])
        self.bn3=nn.BatchNorm1d(mask3.shape[1])
        self.fc4 = nn.Linear(mask3.shape[1] ,out_size)
        self.relu = nn.GELU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.dropout=nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x=self.bn1(x)
        x=self.dropout(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x=self.bn2(x)
        x=self.dropout(x)
        x = self.tanh(x)

        x = self.fc3(x)
        x=self.bn3(x)
        x=self.dropout(x)
        x = self.tanh(x)

        x = self.fc4(x)
        return x

class FullNetwork(nn.Module):
    def __init__(self, out_size=1):
        super(FullNetwork, self).__init__()
        self.fc1 = nn.Linear(mask1.shape[0],mask1.shape[1])
        self.bn1=nn.BatchNorm1d(mask1.shape[1])
        self.fc2 = nn.Linear(mask2.shape[0],mask2.shape[1])
        self.bn2=nn.BatchNorm1d(mask2.shape[1])
        self.fc3 = nn.Linear(mask3.shape[0],mask3.shape[1])
        self.bn3=nn.BatchNorm1d(mask3.shape[1])
        self.fc4 = nn.Linear(mask3.shape[1], out_size)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x=self.bn1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x=self.bn2(x)
        x = self.tanh(x)

        x = self.fc3(x)
        x=self.bn3(x)
        x = self.tanh(x)

        x = self.fc4(x)
        return x