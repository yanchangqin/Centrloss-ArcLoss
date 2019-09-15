import torch
from classify_loss import Cnn_Net
import torchvision
import torch.nn as nn
import torch.utils.data as data
from LOSS import CenterLoss
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt


save_path =r'F:\ycq\centerloss\adm-sum'
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
# print("train_data:",train_data.data.size())
# print("train_label:",train_data.targets.size())

train = data.DataLoader(dataset=train_data,batch_size=100,shuffle=True)
test = data.DataLoader(dataset=test_data,batch_size=1,shuffle=True)
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# net = Net().to(device)
net = Cnn_Net().to(device)
nllloss = nn.NLLLoss(reduction='mean').to(device)
centerloss = CenterLoss(10,2).to(device)
# net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
optimizer1 = optim.Adam(net.parameters())
optimizer2 = optim.Adam(centerloss.parameters())
# optimizer1 = optim.SGD(net.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
# optimizer2 = optim.SGD(centerloss.parameters(), lr =0.005)
x_j = []
loss_ = []
num = 0
def trainer(train_loader,net,epoch,num):
    # print(num)
    print("training Epoch: {}".format(epoch))
    net.train()
    total_output = []
    total_target = []
    total_feature = []

    for j,(x,y) in enumerate(train_loader):
        # x=x.view(-1,28*28)
        x=x.to(device)
        y = y.to(device)
        features,outputs = net(x)
        # y = torch.zeros(y.size(0),10).scatter_(1,y.view(-1,1),1)
        # loss = net.get_loss(outputs,features,y)
        loss1 =nllloss(outputs, y)
        loss2 =centerloss(features, y)
        loss = loss1 +  loss2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        if j %100==0:
            if epoch%20==0:
                if not os.path.exists(os.path.join(save_path, str(epoch))):
                    os.makedirs(os.path.join(save_path, str(epoch)))
                torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))
            print('epoch',epoch,'loss:', loss.item(),'loss1:',loss1.item(),'loss2:',loss2.item())

        # x_j.append(num)
        # num += 1
        #
        # loss_.append(loss)
        #
        total_output.append(outputs)
        total_target.append(y)
        total_feature.append(features)

    # total_output = torch.cat(total_output, dim=0)
    # total_loss =torch.cat(loss_,dim=0)
    total_target = torch.cat(total_target, dim=0)
    total_feature = torch.cat(total_feature, dim=0)
        # print(total_target.shape)
        # print(total_output.shape)
        # print(total_output == total_target)
        # precision = torch.sum(total_output == total_target).numpy() / float(total_target.shape[0])
        # print("Validation accuracy: {}%".format(precision * 100))
    # scatter(total_feature, total_target, epoch, x,x_j,total_loss)
    scatter(total_feature, total_target, epoch, x)
    return num

def tester(test_loader,net,epoch):
    print("Predicting Epoch: {}".format(epoch))
    net.eval()
    total_output = []
    total_target = []
    total_feature = []
    for j, (x, y) in enumerate(test_loader):
        # x = x.view(-1, 28 * 28)
        x = x.to(device)
        y = y.to(device)
        features, outputs = net(x)
        # print('features',features[0].detach().numpy())
        outputs = outputs.argmax(dim=1)
        total_output.append(outputs)
        total_target.append(y)
        total_feature.append(features)
        # print(total_feature)
    total_output = torch.cat(total_output,dim=0)
    total_target = torch.cat(total_target,dim=0)
    total_feature = torch.cat(total_feature,dim=0)
    # print(total_target.shape)
    # print(total_output.shape)
    # print(total_output == total_target)
    precision = torch.sum(total_output == total_target).cpu().numpy() / float(total_target.shape[0])
    print("Validation accuracy: {}%".format(precision * 100))
    scatter(total_feature, total_target, epoch,x)


# def scatter(features,labels,epoch,x,num,loss):
def scatter(features, labels, epoch, x):
    # loss = loss.cpu().detach().numpy()
    # plt.ion()
    plt.clf()
    # fig = plt.figure(constrained_layout=True)

    color_list = ['#800080','#FF7F50','#0000FF','#00FFFF','#696969','#ADFF2F','#FFC0CB','#FF1493','#FFA500','#FF0000']
    for i in range(10):
        plt.plot(features[labels==i,0].cpu().detach().numpy(),features[labels==i,1].cpu().detach().numpy(),'.',c=color_list[i])

        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    for j in range(10):
        coord=(features[labels==j,:].cpu().detach().numpy())
        center_coord=np.mean(coord,axis=0)
        # print(center_coord)
        plt.text(center_coord[0],center_coord[1],j)
        plt.savefig('{}/{}.jpg'.format(save_path,epoch))
        # plt.xlim(xmin=-8, xmax=8)
        # plt.ylim(ymin=-8, ymax=8)
        plt.text(-3, 3, "epoch=%d" % epoch)
    #     plt.pause(0.1)
    # plt.ioff()

for epoch in range(151):
    num = trainer(train,net,epoch,num)