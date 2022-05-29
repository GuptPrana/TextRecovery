import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.transforms.transforms import ColorJitter
from tqdm import tqdm
from emnist_model import Net
from torch.utils.tensorboard import SummaryWriter

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: EMNIST is in torchvision
train_batch = 64
test_batch = 256
data_train = torchvision.datasets.EMNIST('/files/', split='byclass', train=True, download=True,
                            transform=transforms.Compose([
                                # transforms.ColorJitter((0.75, 1.25),(0.75, 1.25),(0.75, 1.25)),
                                transforms.RandomAffine(10, shear=10),
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=0.5, std=0.5)
                            ]))
data_test = torchvision.datasets.EMNIST('/files/', split='byclass', train=False, download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=0.5, std=0.5)
                            ]))
train_loader = data.DataLoader(dataset=data_train, batch_size=train_batch, shuffle=True, num_workers=8)
test_loader = data.DataLoader(dataset=data_test, batch_size=test_batch, shuffle=True, num_workers=8)

# Model 
model = Net().to(device)
# model.load_state_dict(torch.load("model_emnist.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

max_epoch = 50
criterion = nn.CrossEntropyLoss().to(device)
writer = SummaryWriter()

# Train
for this_epoch in range(max_epoch):
    print("Epoch: {}".format(this_epoch+1))
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        images = data.to(device)
        labels = target.to(device)# -1
        output = model(images)# .softmax(dim=1)
        # loss = F.nll_loss(output, labels)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        train_acc += torch.sum(pred == labels)
    train_loss /= len(data_train)
    train_acc /= len(data_train)
    print("train loss {}".format(train_loss))
    print("train accuracy {}".format(train_acc))
    writer.add_scalar("Train Loss", train_loss, this_epoch)
    writer.add_scalar("Train Accuracy", train_acc, this_epoch)

    if (this_epoch+1)%10 == 0:
        torch.save(model.state_dict(), "model_emnist.pth")

    # Test
    print("Testing Model")
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            images = data.to(device)
            labels = target.to(device)
            test_output = model(images)
            test_pred = torch.argmax(test_output, dim=1)
            test_acc += torch.sum(test_pred == labels)
        # print(test_acc)
        test_acc /= len(data_test)
        print("test accuracy {}".format(test_acc))
        writer.add_scalar("Test_Accuracy", test_acc, this_epoch)

writer.flush()
writer.close()
