import torch
import torch.nn as nn
from inst_embedding import InstEmbed, EmbedConfig

class TransConfig():
    def __init__(self, instlength:int, instfeatures:int, nheads:int, nhiddens:int, nlayers:int) -> None:
        self.instlength = instlength
        self.instfeatures = instfeatures
        self.nheads = nheads
        self.nhiddens = nhiddens
        self.nlayers = nlayers
        

class PositionalEncoding(nn.Module):
    def __init__(self, Config:TransConfig, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        
        # 生成位置编码矩阵
        pe = torch.zeros(1, Config.instlength, Config.instfeatures)
        # print(pe.shape)
        position = torch.arange(Config.instlength, dtype=torch.float).unsqueeze(1)
        # print(position.shape)
        div_term = torch.exp(torch.arange(0, Config.instfeatures, 2).float() * (-torch.log(torch.tensor(10000.0)) / Config.instfeatures))
        # print(div_term.shape)
        # print((position*div_term).shape)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)   

class TAO(nn.Module):
    def __init__(self, EmbedConfig, TransConfig, dropout=0.1) -> None:
        super().__init__() 
        self.InstEmbed = InstEmbed(EmbedConfig)
        self.PositionalEncoder = PositionalEncoding(TransConfig)
        
        self.EncoderLayers = nn.TransformerEncoderLayer(d_model=TransConfig.instfeatures,
                                                        nhead=TransConfig.nheads,
                                                        dim_feedforward=TransConfig.nhiddens,
                                                        dropout=dropout,
                                                        batch_first=True
                                                        )
        self.Encoder = nn.TransformerEncoder(self.EncoderLayers, TransConfig.nlayers)
        
        self.linear = nn.Linear(TransConfig.instfeatures * TransConfig.instlength, 2)
    
    def forward(self, x):   #torch.Size([1024, 96, 484])
        x = self.InstEmbed(x)   #torch.Size([1024, 96, 512])
        x = self.PositionalEncoder(x)   #torch.Size([1024, 96, 512])
        x = self.Encoder(x)     #torch.Size([1024, 96, 512])
        x = x.reshape(x.shape[0], -1)   
        x = self.linear(x)      #torch.Size([1024, 2])
        return x
    
# embedconfig = EmbedConfig(74+39, 51, 256, 64, 128, 128, 512, 256, 512)
# transconfig = TransConfig(96, 512, 4, 2048, 4)
# model = TAO(embedconfig, transconfig)

# # print(model)
# input = torch.randn(512, 96, 484)
# model.cuda() 
# input = input.to("cuda")

# output = model(input)
# print(output.shape)       