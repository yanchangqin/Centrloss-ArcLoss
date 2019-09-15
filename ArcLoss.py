import torch
import torch.nn as nn
import torch.nn.functional as F

class Arc_Loss(nn.Module):
    def __init__(self,feature_num,cls_num,m =0.1,s=64):
        super().__init__()
        self.s = s
        self.m = m
        self.W=nn.Parameter(torch.randn(feature_num,cls_num))

    def forward(self, feature):
        _w = F.normalize(self.W,dim=0)
        _x = F.normalize(feature,dim=1)
        cosa = (torch.matmul(_x,_w)/10)
        a = torch.acos(cosa)

        top = torch.exp(torch.cos(a+self.m)*self.s)
        _top = torch.exp(torch.cos(a)*self.s)
        bottom = torch.sum(_top,dim=1,keepdim=True)

        # sina = torch.sqrt(1-torch.pow(cosa,2))
        # cosm = torch.cos(torch.tensor(self.m)).cuda()
        # sinm = torch.cos(torch.tensor(self.m)).cuda()
        # cosa_m =cosa*cosm-sina*sinm
        # top =torch.exp(cosa_m*self.s)
        # _top =torch.exp(cosa*self.s)
        # bottom =torch.sum(_top,dim=1,keepdim=True)

        return (top / (bottom - _top + top))+1e-10

# import math
# class Arc_Loss(nn.Module):
#     def __init__(self, m=0.5, s=10, easy_margin=False, emb_size=10,num_classes=2):
#         super(Arc_Loss, self).__init__()
#
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
#         # num_classes 训练集中总的人脸分类数
#         # emb_size 特征向量长度
#         nn.init.xavier_uniform_(self.weight)
#         # 使用均匀分布来初始化weight
#
#         self.easy_margin = easy_margin
#         self.m = m
#         # 夹角差值 0.5 公式中的m
#         self.s = s
#         # 半径 64 公式中的s
#         # 二者大小都是论文中推荐值
#
#         self.cos_m = math.cos(self.m)
#         self.sin_m = math.sin(self.m)
#         # 差值的cos和sin
#         self.th = math.cos(math.pi - self.m)
#         # 阈值，避免theta + m >= pi
#         self.mm = math.sin(math.pi - self.m) * self.m
#
#     def forward(self, input, label):
#         label =label.cpu()
#         x = F.normalize(input)
#
#         W = F.normalize(self.weight).t()
#         # print(W.size())
#         # 正则化
#         cosine = (F.linear(x, W)).cpu()
#         # cos值
#         sine = torch.sqrt(1.0 - torch.pow(cosine/10, 2))
#         # sin
#         phi = (cosine * self.cos_m - sine * self.sin_m).cpu()
#         # cos(theta + m) 余弦公式
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#             # 如果使用easy_margin
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         one_hot = torch.zeros(cosine.size())
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # print(one_hot)
#         # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         print(output)
#         # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
#         # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
#         # 再乘以半径，经过交叉熵，正好是ArcFace的公式
#         output *= self.s
#         loss = F.cross_entropy(cosine,label)
#
#         # 乘以半径
#         return output,loss



