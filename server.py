from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
from tqdm import tqdm
import uuid
class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):

        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.train_df)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.BatchNorm2d(96),
            #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        
  
  
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
net = SiameseNetwork()
net.load_state_dict(torch.load(r'D:\Signature-Verification\newmodel.pt'))
IMAGEDIR = 'imagesnew/'
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('index.html')
@app.post("/uploadProfilePicture")
async def UploadImage(mypic1:UploadFile=File(...), mypic2:UploadFile=File(...)):
    mypic1.filename = f"{uuid.uuid4()}.png"
    mypic2.filename = f"{uuid.uuid4()}.png"
    
    contents1 = await mypic1.read()
    with open(f"{IMAGEDIR}{mypic1.filename}","wb") as f:
        f.write(contents1)
    contents2 = await mypic2.read()
    with open(f"{IMAGEDIR}{mypic2.filename}","wb") as f:
        f.write(contents2)
    image1_data = Image.open('imagesnew/' + mypic1.filename)
    image1_data = image1_data.convert("L")
    image2_data = Image.open('imagesnew/' + mypic2.filename)
    image2_data = image1_data.convert("L")
    transform=transforms.Compose(
        [transforms.Resize((105, 105)), transforms.ToTensor()]
    )
    image1_data = transform(image1_data)
    image1_data = image1_data.unsqueeze(0)
    image2_data = transform(image2_data)
    image2_data = image2_data.unsqueeze(0)
    ans = net.forward(image1_data,image2_data)
    print(ans)
    '''# print(ans[0][1])
    #res = ans.detach().numpy()
    #resFlat = res.flatten()
    print(resFlat)
    if resFlat[1] > 0.50:
        return FileResponse('result.html')
    else:
        return FileResponse('forgedresult.html')'''

    newans = ans[0]
    res = newans.detach().numpy()
    resFlat = res.flatten()
    print(resFlat)
    if resFlat[1] > 0.50:
        return FileResponse('result.html')
    else:
        return FileResponse('forgedresult.html')
    print(type(res))
    print(newans)