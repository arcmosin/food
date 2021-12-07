import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import ConcatDataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from tqdm.auto import tqdm

train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

device='cuda' if torch.cuda.is_available() else 'cpu'

def openImage(x):
    return Image.open(x)

batch_size=64

test_set=DatasetFolder("food-11/testing",loader=openImage,extensions="jpg",transform=train_tfm)
test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

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

model=Classifier().to(device)
ckpt=torch.load('mymodel/model.pth',map_location='cpu')
model.load_state_dict(ckpt)

model.eval()
predictions=[]

for batch in tqdm(test_loader):
    input,label=batch
    input, label =input.to(device), label.to(device)
    with torch.no_grad():
        output=model(input)
    predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())

with open("predict.csv","w") as f:
    f.write('ID,Category\n')
    for i,pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")