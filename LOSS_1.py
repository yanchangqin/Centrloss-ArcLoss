import torch
from torch.autograd import Variable


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        # print(self.centers)
        self.use_cuda = False

    def forward(self,feat,y ):
        # if self.use_cuda:
        #     hist = Variable(torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1).cuda()
        #     print('hist1',hist)
        # else:
        # print('y',y)
        # print('feat',feat)
        hist = Variable(torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes)+1)#原本统计类别数量，但为何+1？

        centers_count = hist.index_select(0, y.long())  # 计算每个类别对应的数目#3，3，2
        # print('centers_count', centers_count)

        batch_size = feat.size()[0]#batch_size = 3

        feat = feat.view(batch_size, 1, 1, -1).squeeze()#操作结果与原feat一样，有何作用？
        # print('feat1',feat.size()[0])
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,
                                                                                                    feat.size()[1]))
        centers_pred = self.centers.index_select(0, y.long())
        # print(centers_pred)
        diff = feat - centers_pred

        loss = self.loss_weight * 1 /(diff.pow(2).sum(1) / centers_count).sum().float()
        return loss

    # def cuda(self, device_id=None):
    #     self.use_cuda = True
    #     return self._apply(lambda t: t.cuda(device_id))
xs = torch.Tensor([[1,2],[3,4],[5,6]])
label=torch.LongTensor([0,0,1])
centerloss = CenterLoss(2,2)
centerloss.forward(xs,label)