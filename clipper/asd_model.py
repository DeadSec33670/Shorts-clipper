"""Самодостаточная копия архитектуры Light-ASD (CVPR 2023, MIT-лицензия,
github.com/Junhua-Liao/Light-ASD). Объединены Encoder + Classifier + Model
и добавлена обёртка ASD_Net с FC-слоем оценки. Убран хардкод .cuda()."""
import torch
import torch.nn as nn



class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 1), padding = (1, 0), bias = False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 3), padding = (0, 1), bias = False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), padding = (2, 0), bias = False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 5), padding = (0, 2), bias = False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.last = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 1), padding = (0, 0), bias = False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down = False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), stride = (1, 2, 2), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        else:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.last = nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0), bias = False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        return x


class visual_encoder(nn.Module):
    def __init__(self):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, is_down = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block2 = Visual_Block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block3 = Visual_Block(64, 128)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()     

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B*T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(1, 32)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block3 = Audio_Block(64, 128)

        self.__init_weight()
            
    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.squeeze(2).transpose(1, 2)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class BGRU(nn.Module):
    def __init__(self, channel):
        super(BGRU, self).__init__()

        self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        
        self.gelu = nn.GELU()
        self.__init_weight()

    def forward(self, x):
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        x = self.gelu(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()

import torch
import torch.nn as nn


class ASD_Model(nn.Module):
    def __init__(self):
        super(ASD_Model, self).__init__()
        
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.GRU = BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        x = x1 + x2 
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, 128))
        return x    

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  
        outsV = self.forward_visual_backend(visualEmbed)

        return outsAV, outsV

class _LossAV(nn.Module):
    """Только FC-слой из lossAV — остальное (обучение) не нужно для инференса."""
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(128, 2)


class ASD_Net(nn.Module):
    """Инференс-обёртка: model (энкодеры+GRU) + lossAV.FC (оценка речи).
    Имена подмодулей совпадают с ключами в весах Light-ASD."""
    def __init__(self):
        super().__init__()
        self.model = ASD_Model()
        self.lossAV = _LossAV()
