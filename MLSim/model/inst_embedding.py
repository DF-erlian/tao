import  torch
from torch import nn

class EmbedConfig():
    def __init__(self, opfeatures:int, regfeatures:int, memfeatures:int, branchfeatures:int,
                    ophide:int, reghide:int, memhide:int, branchhide:int,
                    instfeatures:int) -> None:
        self.opfeatures = opfeatures
        self.regfeatures = regfeatures
        self.memfeatures = memfeatures
        self.branchfeatures = branchfeatures
        self.ophide = ophide
        self.reghide = reghide
        self.memhide = memhide
        self.branchhide = branchhide
        self.instfeatures = instfeatures
        
class InstEmbed(nn.Module):
    def __init__(self, Config:EmbedConfig) -> None:
        super().__init__()
        self.Config = Config
        self.OpEmbed = nn.Linear(Config.opfeatures, Config.ophide)
        self.RegEmbed = nn.Linear(Config.regfeatures, Config.reghide)
        self.MemEmbed = nn.Linear(Config.memfeatures, Config.memhide)
        self.BrEmbed = nn.Linear(Config.branchfeatures, Config.branchhide)
        self.instEmbed = nn.Linear(Config.ophide + Config.reghide + Config.memhide + Config.branchhide, Config.instfeatures)
        # self.bn = nn.BatchNorm1d(Config.memhide)
        
    def forward(self, x:torch.Tensor):
        op = x[:, :, :self.Config.opfeatures]
        # print(op.shape, op.device, x.device)
        reg = x[:, :, self.Config.opfeatures : self.Config.opfeatures+self.Config.regfeatures]
        mem = x[:, :, self.Config.opfeatures+self.Config.regfeatures : self.Config.opfeatures+self.Config.regfeatures+self.Config.memfeatures]
        br = x[:, :, self.Config.opfeatures+self.Config.regfeatures+self.Config.memfeatures : self.Config.opfeatures+self.Config.regfeatures+self.Config.memfeatures+self.Config.branchfeatures]
        
        op = self.OpEmbed(op)
        reg = self.RegEmbed(reg)
        mem = self.MemEmbed(mem)
        # mem = self.bn(mem.view(-1, self.Config.memhide)).view(-1, op.shape[1], self.Config.memhide)
        br = self.BrEmbed(br)
        
        x = torch.cat((op, reg, mem, br), dim=2)
        x = self.instEmbed(x)
        return x

# config = EmbedConfig(74+39, 51, 256, 64, 128, 128, 512, 256, 512)
# model = InstEmbed(config)    
# print(model)
# input = torch.randn(1024, 96, 484)
# model.cuda()
# input = input.to("cuda")

# print(input.device)
# output = model(input)

# print(output.shape)
