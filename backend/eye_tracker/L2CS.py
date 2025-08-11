import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import math
import torch.nn.functional as F

NUM_BINS = 90
ANGLE_OFFSET = -180
ANGLE_STEP = 4

transform = transforms.Compose(
    [
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# defining L2CS Model


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # gaze
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze, x


# loading L2CS Model


def load_model():
    # Determine the device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    state_dict = torch.load("models/l2cs_trained.pkl", map_location=device)
    if isinstance(state_dict, L2CS):
        return state_dict
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_batch(yaw_logits, pitch_logits, device=None):
    """
    yaw_logits, pitch_logits: [B, NUM_BINS]
    반환: yaw, pitch: [B] (float32)
    """
    if device is None:
        device = yaw_logits.device
    B, NUM_BINS = yaw_logits.shape
    idx_tensor = torch.arange(NUM_BINS, dtype=torch.float32, device=device).unsqueeze(
        0
    )  # [1, NUM_BINS]
    yaw_probs = F.softmax(yaw_logits, dim=1)  # [B, NUM_BINS]
    pitch_probs = F.softmax(pitch_logits, dim=1)
    # 연속 값 예측
    yaw = torch.sum(yaw_probs * idx_tensor, dim=1) * ANGLE_STEP + ANGLE_OFFSET  # [B]
    pitch = torch.sum(pitch_probs * idx_tensor, dim=1) * ANGLE_STEP + ANGLE_OFFSET

    return yaw, pitch  # 그대로 텐서로 반환

def predict(yaw_logits, pitch_logits, device=None):
    yaw_probs = F.softmax(yaw_logits, dim=1)
    pitch_probs = F.softmax(pitch_logits, dim=1)
    idx_tensor = torch.arange(
        NUM_BINS, dtype=torch.float32, device=yaw_logits.device
    ).unsqueeze(0)
    yaw = torch.sum(yaw_probs * idx_tensor, dim=1) * ANGLE_STEP + ANGLE_OFFSET
    pitch = torch.sum(pitch_probs * idx_tensor, dim=1) * ANGLE_STEP + ANGLE_OFFSET
    if yaw.shape[0] == 1:
        return yaw.item(), pitch.item()
    return yaw, pitch
