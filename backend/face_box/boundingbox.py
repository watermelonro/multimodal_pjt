import torch
import torch.nn as nn
import torchvision.models as models
import os


def create_resnet18_simple():
    """ResNet18 기반 단순 모델"""

    class ResNet18Simple(nn.Module):
        def __init__(self):
            super(ResNet18Simple, self).__init__()
            resnet = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x

        def predict(self, x, verbose=False):
            """YOLO 호환을 위한 predict 메서드"""
            with torch.no_grad():
                # ResNet18은 바운딩박스 좌표를 출력 [B, 4]
                coords = self.forward(x)  # [B, 4] normalized coordinates

                # YOLO 결과 형식으로 변환
                class FakeResult:
                    def __init__(self, coords):
                        self.boxes = [FakeBoxes(coords)]

                class FakeBoxes:
                    def __init__(self, coords):
                        # normalized coords [0-1] → pixel coords
                        # 이미지 크기 가정 (448x448 - model_inference.py에서 리사이즈하는 크기)
                        if coords.numel() > 0:
                            # coords: [x1, y1, x2, y2] normalized
                            x1, y1, x2, y2 = coords[0]  # 첫 번째 배치만
                            # 448 크기로 스케일링
                            self.xyxy = [
                                torch.tensor(
                                    [x1 * 448, y1 * 448, x2 * 448, y2 * 448],
                                    dtype=torch.float32,
                                )
                            ]
                        else:
                            # 빈 결과인 경우 전체 이미지 사용
                            self.xyxy = [
                                torch.tensor([0, 0, 448, 448], dtype=torch.float32)
                            ]

                return [FakeResult(coords)]

    return ResNet18Simple()


def load_model():
    """학습된 바운딩박스 모델 로드"""
    model_path = "models/boundingbox_model.pth"

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return None

    # 모델 생성
    model = create_resnet18_simple()

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # state_dict만 로드
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 직접 state_dict인 경우

    model.eval()
    print("✅ 바운딩박스 모델 로드 완료!")

    return model
