import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaleLayer(nn.Module):
    def __init__(self, num_features):
        super(ScaleLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * self.weight.view(1, -1, 1, 1)

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Create the body structure exactly matching the pretrained weights
        body = []
        # First conv
        body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))  # body.0
        
        # Create the exact structure from the pretrained model
        for i in range(1, 66):  # body.1 to body.65
            if i % 2 == 1:
                # Odd layers are ScaleLayer in the pretrained model
                body.append(ScaleLayer(num_feat))
            else:
                body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        
        # Last layer has different dimensions
        body.append(nn.Conv2d(num_feat, 48, 3, 1, 1))  # body.66
        
        self.body = nn.ModuleList(body)
        
        # Upsampling and final layers
        self.conv_body = nn.Conv2d(48, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        
        # Process body features
        body_feat = feat
        for i, layer in enumerate(self.body):
            body_feat = layer(body_feat)
            if i < len(self.body) - 1:  # Don't apply activation after last conv
                body_feat = self.lrelu(body_feat)
        
        # Process remaining features
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # Upsampling
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = RDB(num_feat, num_grow_ch)
        self.rdb2 = RDB(num_feat, num_grow_ch)
        self.rdb3 = RDB(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

    def load_weights(self, model_path, download=True):
        loadnet = torch.load(model_path, map_location=self.device)
        print("Available keys:", loadnet.keys())
        if 'params' in loadnet:
            print("Model keys:", loadnet['params'].keys())
            self.model.load_state_dict(loadnet['params'], strict=True)
        else:
            self.model.load_state_dict(loadnet)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, img):
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device)
        
        with torch.no_grad():
            output = self.model(img)
        
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output = output.transpose(1, 2, 0)
        return output 