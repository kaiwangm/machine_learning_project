import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetModel(nn.Module):

    def __init__(self, pre_trained=None):
        super(ResNetModel, self).__init__()

        if pre_trained:
            self.image_feature_extractor = torch.load(pre_trained)
        else:
            resnet18 = models.resnet34(pretrained=True)
            resnet18 = list(resnet18.children())[:-1]
            self.image_feature_extractor = nn.Sequential(*resnet18)
            self.linear1 = nn.Linear(512, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)

    def forward(self, img, img_v):

        img_feat = self.image_feature_extractor(img)
        img_feat_v = self.image_feature_extractor(img_v)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)
        img_feat_v = img_feat_v.squeeze(3)
        img_feat_v = img_feat_v.squeeze(2)

        img_feat = torch.nn.functional.relu(self.bn6(self.linear1(img_feat)))
        img_feat_v = self.bn6(self.linear1(img_feat_v))

        final_feat = 0.5*(img_feat + img_feat_v)

        return final_feat


class CM3DRModel(nn.Module):

    def __init__(self, image_feature_extractor, pointcloud_feature_extractor, num_classes, dim_bit=32):
        super(CM3DRModel, self).__init__()
        self.image_feature_extractor = image_feature_extractor
        self.pointcloud_feature_extractor = pointcloud_feature_extractor
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            *[nn.Linear(dim_bit, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])
        self.projecter = nn.Linear(512, dim_bit)

    def forward(self, pt, img, img_v):

        img_feat = self.image_feature_extractor(img, img_v)
        point_feat = self.pointcloud_feature_extractor(pt)

        img_feat = self.projecter(img_feat)
        point_feat = self.projecter(point_feat)

        img_pred = self.classifier(img_feat)
        point_pred = self.classifier(point_feat)

        return img_pred, point_pred, img_feat, point_feat
