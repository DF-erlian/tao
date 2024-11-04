import torch
from torch import nn
from torch.nn import functional



class Config():
    def __init__(self, dm: int, dk: int, head_num: int, layer_num: int, intermediate_size: int, vocab_size: int, max_len: int) -> None:
        self.dm = dm  # 这个是词向量的维度
        self.dk = dk  # 这个是自注意力头的宽度
        self.head_num = head_num  # 这个是自注意力头的个数
        self.layer_num = layer_num  # 这个是模型的层数
        self.intermediate_size = intermediate_size # 这个是中间层的宽度
        self.vocab_size = vocab_size # 这个是词汇表的个数
        self.max_len = max_len # 这个是输入序列最长的长度
    



class Embedding(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()   
        self.emb = nn.Embedding(Config.vocab_size, Config.dm)
        self.pos = nn.Embedding(Config.max_len, Config.dm)        
        self.drop_out = nn.Dropout(p=0.1)
        
    def forward(self, x: torch.Tensor):
        y = torch.LongTensor([[i for i in range(0, x.shape[1])] for _ in range(0, x.shape[0])], device = x.device)
        return self.drop_out(self.emb(x) + self.pos(y))
    

class Embedding2(nn.Module):
    def __init__(self, Config:Config) -> None:
        super().__init__()
        self.pos = nn.Embedding(Config.max_len, Config.dm)
        self.drop_out = nn.Dropout(p=0.1)
        
    def forward(self, x: torch.Tensor):
        # y = torch.LongTensor([[i for i in range(0, x.shape[1])] for _ in range(0, x.shape[0])], device = x.device)
        y = torch.tensor([[i for i in range(0, x.shape[1])] for _ in range(0, x.shape[0])], device = x.device, dtype = torch.long)
        y = self.pos(y)
        return self.drop_out(x + y)


class Attn(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()
        self.q_proj = nn.Linear(in_features=Config.dm, out_features=Config.dk * Config.head_num, bias=False)
        self.k_proj = nn.Linear(in_features=Config.dm, out_features=Config.dk * Config.head_num, bias=False)
        self.v_proj = nn.Linear(in_features=Config.dm, out_features=Config.dk * Config.head_num, bias=False)
        self.o_proj = nn.Linear(in_features=Config.dk * Config.head_num, out_features=Config.dm, bias=False)
        self.drop_out = nn.Dropout(p=0.1)
        self.dm = Config.dm
        self.dk = Config.dk
        self.head_num = Config.head_num
    
    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        len = x.shape[1]
        
        Q = self.q_proj(x).reshape(batch, len, self.head_num, self.dk)  # [batch, len, head_num * dk] -> [batch, len, head_num, dk]
        K = self.k_proj(x).reshape(batch, len, self.head_num, self.dk) 
        V = self.v_proj(x).reshape(batch, len, self.head_num, self.dk)
        
        Q = Q.transpose(1, 2) # [batch, len, head_num, dk] -> [batch, head_num, len, dk]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        A = functional.softmax(torch.matmul(Q, K.transpose(2, 3)) / self.dk**0.5, dim=-1) # [batch, head_num, len, len]
        A = self.drop_out(A)
        A = torch.matmul(A, V) # [batch, head_num, len, dk]
        A = A.transpose(1, 2) # [batch, head_num, len, dk] -> [batch, len, head_num, dk]
        A = A.reshape(batch, len, self.head_num * self.dk)
        return self.o_proj(A)
    
class FFN(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()
        self.u_porj = nn.Linear(Config.dm, Config.intermediate_size, bias=False)
        self.d_proj = nn.Linear(Config.intermediate_size, Config.dm, bias=False)
        
    def forward(self, x: torch.Tensor):
        out = functional.sigmoid(self.u_porj(x))
        out = self.d_proj(out)
        return out
    
class Encoder_Module(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()
        self.Attn = Attn(Config)
        self.FFN = FFN(Config)
        
    def forward(self, x: torch.Tensor):
        x1 = self.Attn(x) + x
        x1 = nn.functional.layer_norm(x1, x1.shape)
        
        y1 = self.FFN(x1) + x1
        y1 = nn.functional.layer_norm(y1, y1.shape)
        return y1

class Out_Head(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()
        self.pooler = nn.Linear(Config.dm, Config.dm, bias=False)
        self.o_head = nn.Linear(Config.dm, 2, bias=False)
        
    def forward(self, x: torch.Tensor):
        x = self.pooler(x)
        x = functional.sigmoid(x)
        x = x[:, 0, :]
        x.reshape(x.shape[0], x.shape[-1])
        x = self.o_head(x)
        return x
        
class TAOModel(nn.Module):
    def __init__(self, Config: Config) -> None:
        super().__init__()
        self.Emb = Embedding2(Config)
        self.Encoder_List = nn.ModuleList([Encoder_Module(Config) for _ in range(Config.layer_num)])
        self.Out = Out_Head(Config)
        
    def forward(self, x):
        x = self.Emb(x)
        for i in range(self.Encoder_List.__len__()):
            x = self.Encoder_List[i](x)
        x = self.Out(x)
        
        return x        

 
        