'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


from models.text_encoder import load_clip_to_cpu, TextEncoder, PromptLearner

from models.clip import clip

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, args = None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.conv1x1 = nn.Conv2d(128, 32, kernel_size = 1)


        classnames = ['benign', 'unsure', 'malignant']
        attrnames = ['malignancy', 'subtlety', 'internalStructure', 'calcification', 
        'sphericity', 'margin', 'lobulation', 'spiculation', 'texture']

        # Text encoder
        clip_model = load_clip_to_cpu()
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder(clip_model, args)
        self.prompt_learner = PromptLearner(classnames, attrnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_attr = self.prompt_learner.tokenized_attr


    def forward(self, x,
        args, 
        label = None,
        weights = None, 
        pre_text_tensors = None, 
        temp = None,
        label_distribution = None, ood_test=False, mode = 'train'):


        tokenized_prompts = self.tokenized_prompts
        tokenized_attr = self.tokenized_attr
        logit_scale = self.logit_scale.exp()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.shape[0], 128, 8, 8)
        out = self.conv1x1(out)

        group_out = torch.split(out, 8, dim = 1)
        group_out = torch.cat([i.view(i.shape[0], -1).unsqueeze(1) for i in group_out], dim = 1) # (bs, 4, 512)

        image_features = group_out # (bs, n_ctx, 512)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if args.language :
            out = F.adaptive_avg_pool2d(out, 4)
            out = self.linear(out.view(out.shape[0], -1))
            prompts, attr_prompts = self.prompt_learner(image_features)
            if mode == 'train':
                logits = []
                ca_logits = []
                ia_logits = []
                for batch_i, (pts_i, attr_pts_i, imf_i, label_i, weights_i) in enumerate(zip(
                            prompts, attr_prompts, image_features, label, weights)):

                    weights_i = F.softmax(weights_i, dim = -1)

                    # image --- class prompts (IC)
                    text_feature = self.text_encoder(pts_i, tokenized_prompts)
                    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                    l_i = logit_scale * imf_i @ text_feature.t()
                    l_i = F.cross_entropy(l_i, torch.cat([label[j].unsqueeze(0) for j in range(l_i.shape[0])], dim = 0))

                    # class prompts --- attribute prompts (CA)
                    attr_feature = self.text_encoder(attr_pts_i, tokenized_attr) # (8, 512)
                    attr_feature = attr_feature / attr_feature.norm(dim = -1, keepdim = True)
                    ca_l_i = logit_scale * text_feature @ attr_feature.t() # (n_class, 8)
                    ca_l_i = ca_l_i * weights_i.unsqueeze(0)

                    # image --- attribute prompts (IA)
                    ia_l_i = logit_scale * imf_i @ attr_feature.t() # (n_ctx, 8)
                    ia_l_i = ia_l_i * weights_i.unsqueeze(0)
                   
                    ia_logits.append(ia_l_i.mean(1))
                    logits.append(l_i)
                    ca_logits.append(ca_l_i.mean(1))

                logits = torch.stack(logits)
                ca_logits = torch.stack(ca_logits)
                ia_logits = torch.stack(ia_logits)

                contrast_loss = torch.mean(logits)
                ca_contrast_loss = F.cross_entropy(ca_logits, label)
                ia_contrast_loss = F.cross_entropy(ia_logits, label)

                return out, 1*contrast_loss + 0.5*ca_contrast_loss + 1*ia_contrast_loss
            else:
                return out, 0


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)



def ResNet18(num_classes, args):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, args = args)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

