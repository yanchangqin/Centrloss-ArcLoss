import torch
import torch.nn as nn
import torch.nn.functional as f

class CenterLoss(nn.Module):
    def __init__(self,cls_num,feature_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self, xs,label):
        xs = f.normalize(xs)
        #根据label索引选择中心点
        cen_select = self.center.index_select(dim=0,index=label)
        #统计出每个类的data---->[2,1]
        count = torch.histc(label.float(),bins=10,min=0,max=9)
        #根据count出来的数量从label里重新选择，count_dis为每个data对于的数量----》[2,2,1]
        count_dis = count.index_select(dim=0,index=label)

        return torch.sum(torch.sum((xs-cen_select)**2,dim =1)/count_dis.float())

#
# xs = torch.Tensor([[1,2],[3,4],[5,6]])
# label=torch.LongTensor([0,0,1])
# centerloss = CenterLoss(2,2)
# centerloss.forward(xs,label)

