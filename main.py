import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset,DataLoader,Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'

train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def openImage(x):
    return Image.open(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4,4,0)
        )

        self.fc_layers=nn.Sequential(
            nn.Linear(256*8*8,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,11)
        )

    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

batch_size=64
train_set=DatasetFolder("food-11/training/labeled",loader=openImage,extensions='jpg',transform=train_tfm)
valid_set=DatasetFolder("food-11/validation",loader=openImage,extensions='jpg',transform=train_tfm)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
valid_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)

model=Classifier().to(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0003,alpha=0.99,eps=1e-08,
#                                weight_decay=0,momentum=0,centered=False)
#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, \
#            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


n_epoch=40

#trainlossForMatplot=np.array([])
trainlossForMatplot=[]
trainaccForMatplot=[]
#validlossForMatplot=np.array([])
validlossForMatplot=[]
validaccForMatplot=[]
x=[]
min_loss=1000
if __name__=='__main__':
    for epoch in range(n_epoch):
        train_loss = []
        train_accs = []
        valid_loss = []
        valid_accs = []
        model.train()
        for batch in tqdm(train_loader):
            input,labels = batch
            input, labels=input.to(device),labels.to(device)
            output=model(input)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm(model.parameters(),max_norm=10)
            optimizer.step()
            acc = (output.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc.item())

        trainloss=sum(train_loss)/len(train_loss)
        trainacc=sum(train_accs)/len(train_accs)

        trainlossForMatplot.append(trainloss)
        trainaccForMatplot.append( trainacc)
        print(f"[Train | {epoch + 1:03d}/{n_epoch:03d}] loss = {trainloss:.5f},acc={trainacc:5f}")


        model.eval()
        for batch in tqdm(valid_loader):
            input,labels = batch
            input, labels = input.to(device), labels.to(device)
            with torch.no_grad():
                output=model(input)
            loss = criterion(output,labels)
            acc = (output.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())
#        scheduler.step(trainloss)

        validloss = sum(valid_loss)/len(valid_loss)
        validacc = sum(valid_accs)/len(valid_accs)
        validlossForMatplot.append(validloss)
        validaccForMatplot.append(validacc)
        x.append(epoch)
        plt.ion()#交互模式,与plt.ioff()配套使用;使得plt.plot()或plt.imshow()直接输出图像,不需要show();
                # 即使遇到show(),程序也会继续执行下去,但需要在show()前加上plt.ioff();
        plt.clf()# Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
        ax = plt.gca()#获得当前Axes对象
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))#设置x轴刻度间隔为1
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.xlim(-0.5,epoch+0.5)#设置x轴范围为-0.5到epoch+0.5,+0.5是为了留白
        plt.title('loss in train & valid')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.plot(x, trainlossForMatplot,x, validlossForMatplot)
        plt.legend(['train','valid'])#设置图例
        plt.text(epoch+0.5,trainlossForMatplot[0],f"{trainloss:0.4f}", fontsize=10,style='italic',color='#1f77b4')
        plt.text(epoch+0.5,validloss,f"{validloss:0.4f}", fontsize=10,style='italic',color='#ff7f0e')#设置注释
        plt.pause(0.1)
        plt.ioff()#关闭交互模式


        if (validloss<=0.2) & (validloss<=min_loss):
            min_loss=validloss
            print(f'Save model (epoch = {epoch},loss={validloss},acc={validacc})')
            torch.save(model.state_dict(),f'mymodel/model{epoch}.pth')

        np.savetxt('trainLossData.csv',trainlossForMatplot,delimiter=',')
        np.savetxt('trainAccData.csv', trainaccForMatplot, delimiter=',')
        np.savetxt('validLossData.csv', validlossForMatplot, delimiter=',')
        np.savetxt('validAccData.csv', validaccForMatplot, delimiter=',')

