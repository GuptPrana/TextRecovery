import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    
    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class rcnn(nn.Module):
    def __init__(self):
        super(rcnn, self).__init__()

        kernel = [3, 3, 3, 3, 3, 3, 2]
        padding = [1, 1, 1, 1, 1, 1, 0]
        stride = [1, 1, 1, 1, 1, 1, 1]
        channels = [64, 128, 256, 256, 512, 512, 512]

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else channels[i - 1]
            nOut = channels[i]
            layer = nn.Sequential()
            layer.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel[i], stride[i], padding[i]))
            if batchNormalization:
                layer.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            layer.add_module('relu{0}'.format(i), nn.ReLU(True))

        cnn = nn.Sequential(
            convRelu(0),
            nn.MaxPool2d(2, 2),  # 64x16x64
            convRelu(1),
            nn.MaxPool2d(2, 2),  # 128x8x32
            convRelu(2, True),
            convRelu(3),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x16
            convRelu(4, True),
            convRelu(5),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x16
            convRelu(6, True))  # 512x1x16

        self.cnn = cnn
        self.lstm = nn.Sequential(
            BidirectionalLSTM(256*2, 256, 256),
            BidirectionalLSTM(256, 256, 85))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.lstm(conv)
        output = output.transpose(1,0) #Tbh to bth
        return output

