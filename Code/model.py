import torch
import torch.nn as nn
import torchvision.models as models


class FusionModelWithAttention(nn.Module):
    def __init__(self, num_keypoints, num_classes, dropout_rate=0.1):
        super(FusionModelWithAttention, self).__init__()
        # 图像特征提取器
        self.image_feature_extractor = models.resnet18(pretrained=True)
        self.image_feature_extractor.fc = nn.Identity()  # 移除顶层

        # 关键点特征提取器
        self.keypoints_feature_extractor = nn.Sequential(
            nn.Linear(num_keypoints, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 在第一个全连接层后添加 Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)   # 在第二个全连接层后添加 Dropout
        )

        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)   # 在融合层后添加 Dropout
        )

        # 自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # 在分类器之前添加 Dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, image, keypoints_tensor):
        # 确保关键点张量的形状是正确的
        if keypoints_tensor.dim() > 2:
            keypoints_tensor = keypoints_tensor.view(keypoints_tensor.size(0), -1)

        image_features = self.image_feature_extractor(image)
        keypoints_features = self.keypoints_feature_extractor(keypoints_tensor)

        # 特征融合
        combined_features = torch.cat((image_features, keypoints_features), dim=1)
        fused_features = self.fusion_layer(combined_features)

        attention_output, _ = self.attention(fused_features.unsqueeze(0), fused_features.unsqueeze(0),
                                             fused_features.unsqueeze(0))
        attention_output = attention_output.squeeze(0)

        # 分类
        output = self.classifier(attention_output)
        return output


# model = models.resnet18(pretrained=True)
# # 移除最后的全连接层，获取特征向量
# model.fc = nn.Identity()
#
# # 创建一个虚拟输入（例如，对于期望的输入尺寸为224x224的模型）
# dummy_input = torch.randn(1, 3, 224, 224)
# features = model(dummy_input)
#
# print(features.shape)







    # def forward(self, image, keypoints_tensor):
    #     print(f"Image shape: {image.shape}")  # 打印图像张量的形状
    #     image_features = self.image_feature_extractor(image)
    #
    #     print(f"Keypoints input shape: {keypoints_tensor.shape}")  # 打印输入关键点张量的形状
    #     keypoints_features = self.keypoints_feature_extractor(keypoints_tensor)
    #     print(f"Keypoints features shape: {keypoints_features.shape}")  # 打印关键点特征的形状
    #
    #     # 特征融合
    #     combined_features = torch.cat((image_features, keypoints_features), dim=1)
    #     print(f"Combined features shape: {combined_features.shape}")  # 打印融合后的特征形状
    #     fused_features = self.fusion_layer(combined_features)
    #     print(f"Fused features shape: {fused_features.shape}")  # 打印融合特征后的形状
    #
    #     attention_output, _ = self.attention(fused_features.unsqueeze(0), fused_features.unsqueeze(0),
    #                                          fused_features.unsqueeze(0))
    #     attention_output = attention_output.squeeze(0)
    #
    #     # 分类
    #     output = self.classifier(attention_output)
    #     return output