import torch
import os
import torch.nn as nn
import torchvision
import torch.utils.data as data
from ArcLoss import Arc_Loss
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from LOSS import CenterLoss
import torch.nn.functional as F
import PIL.Image as image
import numpy as np


class Cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnlayer = nn.Sequential(
            nn.Conv2d(1,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 2, padding=1),

            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, 3, 2, padding=1),
        )

        self.linerlayer_hidden = nn.Sequential(
            nn.Linear(16*7*7,256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),

            nn.Linear(64, 2,bias=False)
        )
        self.linerlayer_out =nn.Sequential(
            nn.Linear(2,10)
        )
    def forward(self, x):
        y = self.cnnlayer(x)
        y = y.view(-1, 16 * 7 * 7)
        feature = self.linerlayer_hidden(y)
        output = self.linerlayer_out(feature)
        return feature, F.log_softmax(output, dim=1)



save_path = r'F:\ycq\centerloss\14\100'
save_param = r'param_net.pt'
train_data = torchvision.datasets.MNIST(
    root="MNIST_DATA",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root="MNIST_DATA",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)
train = data.DataLoader(dataset=train_data, batch_size=1024, shuffle=True)

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# net = Net().to(device)
net = Cnn_net().to(device)
nllloss = nn.NLLLoss(reduction='sum').to(device)
centerloss = CenterLoss(10,2).to(device)
arcloss = Arc_Loss(2,10).to(device)
net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
optimizer1 = optim.Adam(net.parameters())

# optimizer2 = optim.Adam(arcloss.parameters())

# optimizer4nn = optim.SGD(net.parameters(),lr=0.00001,momentum=0.9, weight_decay=0.0005)
# sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

def trainer(train_loader, net, epoch):
    print("training Epoch: {}".format(epoch))
    net.train()
    total_output = []
    total_target = []
    total_feature = []
    for j, (x, y) in enumerate(train_loader):
        # x=x.view(-1,28*28)
        x = x.to(device)
        y = y.to(device)
        features,output = net(x)
        # print(features.size())
        # print(output)
        r = 0.5
        out = arcloss(features)
        loss1 = nllloss(out, y)
        loss2 = centerloss(features, y)
        loss = r * loss1 + (1 - r) * loss2

        optimizer1.zero_grad()
        # optimizer4nn.zero_grad()

        # optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        # optimizer2.step()
        # optimizer4nn.step()
        print('epoch', epoch, 'loss:', loss.item())
        # scatter(features, y, epoch, x)
        # if j % 100 == 0:
        if epoch%100==0:
            if not os.path.exists(os.path.join(save_path, str(epoch))):
                os.makedirs(os.path.join(save_path, str(epoch)))
            torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))

        # total_output.append(outputs)
        # total_target.append(y)
        # total_feature.append(features)
    # # total_output = torch.cat(total_output, dim=0)
    # total_target = torch.cat(total_target, dim=0)
    # total_feature = torch.cat(total_feature, dim=0)
    # # print(total_target.shape)
    # # print(total_output.shape)
    # # print(total_output == total_target)
    # # precision = torch.sum(total_output == total_target).numpy() / float(total_target.shape[0])
    # # print("Validation accuracy: {}%".format(precision * 100))
    # scatter(total_feature, total_target, epoch, x)

def scatter(features, labels, epoch,features1):

    # plt.ion()
    plt.clf()
    color_list = ['#800080', '#000000', '#0000FF', '#00FFFF', '#696969', '#ADFF2F', '#FFC0CB', '#FF1493', '#FFA500',
                  '#FF0000']
    for i in range(10):
        plt.plot(features[labels == i, 0].cpu().detach().numpy(), features[labels == i, 1].cpu().detach().numpy(),
                 '.', c=color_list[i])
        # plt.legend(loc='upper left',color_list[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.plot(features1[labels == i, 0].cpu().detach().numpy(), features1[labels == i, 1].cpu().detach().numpy(),
                 '*', c='black')
        # if epoch % 10 == 0:
        # plt.savefig(r'F:\ycq\centerloss\fix\epoch=%d.jpg' % epoch)
    # if not os.path.exists('{0}/{1}.jpg'.format(save_path, str(epoch))):
        plt.savefig('{}/{}.jpg'.format(save_path, epoch))
        # plt.xlim(xmin=-8, xmax=8)
        # plt.ylim(ymin=-8, ymax=8)
        # plt.text(-7.8, 7.3, "epoch=%d" % epoch)
    #     plt.pause(0.1)
    # plt.ioff()
# def tester(epoch,img_path =r'F:\ycq\centerloss\num2',path=r'F:\ycq\centerloss\num3'):
#     print("Predicting Epoch: {}".format(epoch))
#     net.eval()
#     total_output = []
#     total_target = []
#     total_feature = []
#     for name in os.listdir(img_path):
#         im =image.open(os.path.join(img_path,name))
#         im = im.convert('L')
#         # im.show()
#         im =im.resize((28,28))
#         array =np.array(im)
#         array =torch.Tensor([array])
#         array = array.unsqueeze(0)
#         array = array.to(device)
#         array = F.normalize(array)
#         features, outputs = net(array)
#         features = features.cpu()
#         outputs = outputs.cpu()
#         outputs = outputs.argmax(dim=1)
#         total_output.append(outputs)
#         # print(total_output)
#         total_target.append(torch.Tensor([int(name[0])]))
#         total_feature.append(features)
#     total_output = torch.cat(total_output, dim=0)
#     # print(total_output)
#     total_target = torch.cat(total_target).long()
#     # print(total_output)
#     # print(total_target)
#     total_feature = torch.cat(total_feature,dim=0)
#     # precision = torch.sum(total_output == total_target).cpu().numpy() / float(total_target.shape[0])
#     # print("Validation accuracy: {}%".format(precision * 100))
#     for name in os.listdir(path):
#         im =image.open(os.path.join(path,name))
#         im = im.convert('L')
#         # im.show()
#         im =im.resize((28,28))
#         array =np.array(im)
#         array =torch.Tensor([array])
#         array = array.unsqueeze(0)
#         array = array.to(device)
#         array =F.normalize(array)
#         features1, outputs = net(array)
#         label =torch.Tensor([int(name[0])])
#     # scatter(total_feature, total_target, epoch,features1)
#     detect(total_feature,total_target)
# def detect(features,labels,path = r'F:\ycq\centerloss\num3'):
#     feature_new = []
#     feature_total =[]
#     # print(features)
#     for name in os.listdir(path):
#         im =image.open(os.path.join(path,name))
#         im = im.convert('L')
#         # im.show()
#         im =im.resize((28,28))
#         array =np.array(im)
#         array =torch.Tensor([array])
#         array = array.unsqueeze(0)
#         array = array.to(device)
#         array =F.normalize(array)
#         features1, outputs = net(array)
#         features1 = features1.cpu()
#         feature_total = features.cpu()
#         # distance = torch.sqrt(torch.sum((features1-feature_total)**2,dim=1))
#         # feature_所有类别的特征向量
#         feature_ = F.normalize(feature_total).cpu()
#         # print(feature_.size())
#         # feature_测试类别的特征向量
#         feature1_ = F.normalize(features1).t().cpu()
#         # print(feature1_.size())
#         cosa = torch.matmul(feature_, feature1_)
#         # max_a = torch.argmin(distance)
#         print(torch.argmax(cosa).numpy())
#         # print(labels[max_a].numpy())
#         exit()


def tester1(img_path =r'F:\ycq\detector_CenterLoss\mnist\o1',path=r'F:\ycq\centerloss'):
    net.eval()
    total_output = []
    total_target = []
    total_feature = []
    for name in os.listdir(img_path):
        for name_ in os.listdir(os.path.join(img_path,name)):
            img_path1 = os.path.join(img_path,name)
            im =image.open(os.path.join(img_path1,name_))
            im = im.convert('L')
            # im.show()
            im =im.resize((28,28))
            array =np.array(im)
            array =torch.Tensor([array])
            array = array.unsqueeze(0)
            array = array.to(device)
            array = F.normalize(array)
            features, outputs = net(array)
            features = features.cpu()
            outputs = outputs.cpu()
            outputs = outputs.argmax(dim=1)
            total_output.append(outputs)
            # print(total_output)
            total_target.append(torch.Tensor([int(name[0])]))
            total_feature.append(features)
    total_output = torch.cat(total_output, dim=0)
    # print(total_output)
    total_target = torch.cat(total_target).long()
    # print(total_output)
    # print(total_target)
    total_feature = torch.cat(total_feature,dim=0)
    print(total_feature.size())
    # precision = torch.sum(total_output == total_target).cpu().numpy() / float(total_target.shape[0])
    # print("Validation accuracy: {}%".format(precision * 100))
    detect(total_feature,total_target)

def detect(features,labels,path = r'F:\ycq\detector_CenterLoss\mnist\test'):
    feature_new = []
    feature_total =[]
    # print(features)
    for name in os.listdir(path):
        im =image.open(os.path.join(path,name))
        im = im.convert('L')
        # im.show()
        im =im.resize((28,28))
        array =np.array(im)
        array =torch.Tensor([array])
        array = array.unsqueeze(0)
        array = array.to(device)
        array =F.normalize(array)
        features1, outputs = net(array)
        features1 = features1.cpu()
        feature_total = features.cpu()
        # distance = torch.sqrt(torch.sum((features1-feature_total)**2,dim=1))
        # feature_所有类别的特征向量
        feature_ = F.normalize(feature_total).cpu()
        # print(feature_.size())
        # feature_测试类别的特征向量
        feature1_ = F.normalize(features1).t().cpu()
        # print(feature1_.size())
        cosa = torch.matmul(feature_, feature1_)
        # print(cosa)
        # exit()
        # max_a = torch.argmin(distance)
        print(torch.argmax(cosa).numpy()//9)
        # print(labels[max_a].numpy())
        exit()
for epoch in range(101):
    # trainer(train, net, epoch)
    tester1()


