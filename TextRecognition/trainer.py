import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

from dataset import trdg_dataset
from loss import ctcloss
from model import rcnn
from converter import OCRLabelConverter

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = rcnn()
model = model.to(device)

# Dataset
trainset = trdg_dataset(split="train")
valset = trdg_dataset(split="val")
# Iterable loader
batch_size = 32
num_workers = 8
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=True, pin_memory=True,
                                         num_workers=num_workers)

# Loss Function
criterion = ctcloss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)

# Num_epochs
num_epochs = 50

# Log
writer = SummaryWriter()

# Training 
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 20)
    print("Training")
    model.train()
    running_loss = 0.0
    for batch_idx, data in tqdm(enumerate(trainloader)):
        optimizer.zero_grad() 
        images = data['image'].to(device)
        labels = data['label'].to(device)
        pred = model(images)
        targets, target_size = OCRLabelConverter.encode(labels)
        pred = pred.contiguous()#.cpu()
        pred = torch.nn.functional.log_softmax(pred, 2)
        T, B, H = pred.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets= targets.view(-1).contiguous()#.to(device)
        loss = criterion(pred, targets, pred_sizes, target_size)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # running_acc += compute_acc(pred, labels)
    running_loss = running_loss/len(trainset)
    scheduler.step()
    # log
    writer.add_scalar("TrainCTCLoss/Epoch", running_loss, epoch)
    print("=> Epoch:{} - train loss: {:.4f}".format(epoch, running_loss))
    if epoch % 20 == 0:
        torch.save(model.state_dict(), "model.pth")
        torch.save(optimizer.state_dict(), "optimizer.pth")
        torch.save(scheduler.state_dict(), "scheduler.pth")  
    # Sampling Outputs
    # if (epoch+1)%10:
    #   print(OCRLabelConverter.decode(pred))
        image = images.detach().cpu().numpy()
        label = labels.detach().cpu().numpy()
        for i in range(3):
            keepim = image[i]
            keepla = label[i]
            for index in range(16):
                for j in range(64):
                    keepim[index][j] *= 25.5
            print("label {}".format(keepla))                  
            im = Image.fromarray(np.int8(imageslice)).convert('L')
            impath = "./image/" + str(epoch) + "_" + str(i) + ".png"
            im.save(impath)
    #   

    '''
    # Validation 
    print("Validation")
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for batch_idx, data in tqdm(enumerate(valloader)):
            if batch_idx == 100:
                break
            optimizer.zero_grad() 
            images = data['image'].to(device)
            labels = data['label'].to(device)
            pred = model(images)
            targets, target_size = OCRLabelConverter.encode(labels)
            pred = pred.contiguous()#.cpu()
            pred = torch.nn.functional.log_softmax(pred, 2)
            T, B, H = pred.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            targets= targets.view(-1).contiguous()#.to(device)
            loss = criterion(pred, targets, pred_sizes, target_size)
            running_loss += loss.item()
        # log
        running_loss /= (batch_size*100)
        writer.add_scalar("ValCTCLoss/Epoch", running_loss, epoch)
        print("=> Epoch:{} - val loss: {:.4f}".format(epoch, running_loss))
        
    '''    

# Tensorboard
writer.flush()
writer.close()




