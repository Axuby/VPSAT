import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_encoder import VpSatNetTransformer, VanishingPointPredictionHead, SimpleVPHead


def multi_vp_loss(predictions, targets):
    """
    Compute the loss for all 3 vanishing points.

    Args:
        predictions (torch.Tensor): Predicted VPs, shape [batch_size, 3, 3].
        targets (torch.Tensor): Ground-truth VPs, shape [batch_size, 3, 3].
    Returns:
        torch.Tensor: Loss value.
    """
    predictions = F.normalize(predictions, p=2, dim=-1)
    targets = F.normalize(targets, p=2, dim=-1)
    cosine_sim = torch.sum(predictions * targets, dim=-1)  # Shape: [batch_size, 3]
    loss = 1 - cosine_sim.mean(dim=-1)
    return loss.mean()  # Average across batch

class FeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),  # Valid depthwise conv
            nn.Conv2d(3, 64, kernel_size=1),  # Pointwise conv to expand channels
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=64),  # Downsampling to 16x16
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=128),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_projection = nn.Linear(256, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # Downsampling to 16x16 happens here
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)  # GAP reduces spatial dimensions
        x = x.view(x.size(0), -1)  # Flatten
        return x


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super(ResNetFeatureExtractor, self).__init__()

        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        # removes the classification layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(512, d_model)  # ResNet18 outputs 512 channels

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.projection(x)
        return x


def get_feature_extractor(extractor_type, d_model):
    if extractor_type == 'cnn':
        return FeatureExtractor(d_model)
    elif extractor_type == 'resnet':
        return ResNetFeatureExtractor(d_model)
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}")

def get_vp_head(head_type, d_model, num_vpts=3):
    if head_type == 'simple':
        return SimpleVPHead(d_model, num_vpts)
    elif head_type == 'hybrid':
        return VanishingPointPredictionHead(d_model, num_vpts)
    else:
        raise ValueError(f"Unknown VP head type: {head_type}")

class VpSatNet(nn.Module):
    def __init__(self, C):
        super(VpSatNet, self).__init__()
        extractor_type = C.model.get('feature_extractor_type', 'cnn')
        self.feature_extractor = get_feature_extractor(
            extractor_type,
            C.model.transformer.d_model
        )
        self.feature_projection = nn.Linear(256, C.model.transformer.d_model)
        self.transformer_encoder = VpSatNetTransformer(C.model.transformer)
        # self.vp_head = VanishingPointPredictionHead(
        #     C.model.transformer.d_model, num_vpts=C.io.num_vpts
        # )
        vp_head_type = C.model.get('vp_head_type', 'simple')
        self.vp_head = get_vp_head(
            vp_head_type,
            C.model.transformer.d_model,
            C.io.num_vpts
        )

    def forward(self, image_patches, vpts=None, line_segments=None, mode="train"):
        features = self.feature_extractor(image_patches)
        if len(features.shape) == 2:  # [batch_size, features]
            features = features.unsqueeze(1)  # [batch_size, 1, features]
        features = features.view(features.size(0), -1, features.size(1))
        features = self.feature_projection(features)

        if line_segments is not None:
            combined_features = torch.cat((features, line_segments), dim=1)
            transformer_output = self.transformer_encoder(combined_features)
        else:
            transformer_output = self.transformer_encoder(features)

        vp_prediction = self.vp_head(transformer_output)
        return vp_prediction
