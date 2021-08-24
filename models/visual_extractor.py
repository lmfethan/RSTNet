import torch
import torch.nn as nn
import torchvision.models as models
from .SimAM import simam_module

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # model.layer1.add_module('simam1', simam_module())
        # model.layer2.add_module('simam2', simam_module())
        # model.layer3.add_module('simam3', simam_module())
        # model.layer4.add_module('simam4', simam_module())
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        c, h, w = images.shape[-3:]
        bs = images.shape[0]
        iu_xray = False
        if len(images.shape) == 5: # iu_xray
            images = images.view(-1, c, h, w).contiguous()
            iu_xray = True
        patch_feats = self.model(images)
        patch_feats = patch_feats.reshape(patch_feats.shape[0], patch_feats.shape[1], -1).permute(0, 2, 1).contiguous()
        if iu_xray:
            patch_feats = patch_feats.view(bs, 2, patch_feats.shape[-2], patch_feats.shape[-1])
            patch_feats = torch.cat((patch_feats[:, 0], patch_feats[:, 1]), dim=1)
        return patch_feats
