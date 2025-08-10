import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseClassifier(nn.Module):
    def __init__(
        self, input_size=25, hidden_sizes=[128, 512, 256], num_classes=6, dropout=0.3
    ):
        super(NoiseClassifier, self).__init__()

        layers = []
        prev_size = input_size  # 25 유지

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x, return_features=True):
        print(f"🔊 noise_classifier 입력 차원: {x.shape}")

        # [1, 1, 96, 64] → [1, 25]로 변환
        if x.dim() == 4:  # [B, C, H, W]
            # Adaptive pooling으로 [96, 64] → [5, 5] → flatten
            x = F.adaptive_avg_pool2d(x, (5, 5))  # [1, 1, 5, 5]
            x = x.flatten(1)  # [1, 25]
        elif x.dim() == 3:
            x = x.flatten(1)

        features = self.feature_extractor(x)
        print(f"🔊 audio_feature 출력 차원: {features.shape}")

        out = self.classifier(features)

        if return_features:
            return features, out
        return out


# load_model은 그대로 유지


def load_model():
    model = NoiseClassifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(
        "models/audio_noise_classifier.pth", map_location=torch.device(device)
    )
    if isinstance(state_dict, NoiseClassifier):
        return state_dict
    model.load_state_dict(state_dict)
    model.eval()
    return model
