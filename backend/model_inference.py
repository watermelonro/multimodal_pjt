import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from eye_tracker.L2CS import load_model as l2cs_load
from eye_tracker.L2CS import predict
from noise_classifier.classifier import load_model as noise_classifier_load
from face_box.yolo import load_model as face_box_load
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import face_alignment
import warnings
import logging

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 워닝 무시 (tqdm 깨짐 방지)
warnings.filterwarnings("ignore")

NUM_BINS = 90
ANGLE_OFFSET = -180
ANGLE_STEP = 4


class MobileNetModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(MobileNetModel, self).__init__()

        # MobileNetV2 backbone (pre-trained)
        self.mobilenet = mobilenet_v2(pretrained=True)

        # 마지막 classifier 제거
        self.backbone = self.mobilenet.features

        # AvgPool for feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)  # Flatten
        return features  # 1024-dimensional feature vector


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        x = self.projector(x)
        return F.normalize(x, dim=1)


class FinalClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=5):
        super(FinalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerFusion, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, visual_embed, audio_embed):
        """
        patch_embed: (B, N, 128) - 이미지 패치 임베딩
        visual_embed: (B, 3, 128) - 시각적 임베딩
        audio_embed: (B, 128) - 오디오 임베딩
        """
        B = visual_embed.size(0)

        # visual_embed, audio_embed: (B, 128) → (B, 1, 128)
        audio_embed = audio_embed.unsqueeze(1)

        # [CLS] token 생성: (B, 1, 128)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 전체 시퀀스 생성: (B, 3, 128)
        fusion_input = torch.cat([cls_tokens, visual_embed, audio_embed], dim=1)  # (B, N+3, 128)

        # Transformer Encoder
        fusion_output = self.transformer(fusion_input)  # (B, 3, 128)

        # [CLS] token 출력만 추출 (fused representation)
        return fusion_output[:, 0]  # (B, 128)


class EndtoEndModel(nn.Module):
    def __init__(self, mobile_net, eye_track_model, noise_classifier, face_mesh_model,
                 mobilenet_proj, l2cs_proj, fa_proj, audio_proj,
                 fusion_block, final_classifier):
        super(EndtoEndModel, self).__init__()
        self.mobile_net = mobile_net
        self.eye_track_model = eye_track_model
        self.noise_classifier = noise_classifier
        self.face_mesh_model = face_mesh_model  # → face_alignment.FaceAlignment 객체
        self.mobilenet_proj = mobilenet_proj
        self.l2cs_proj = l2cs_proj
        self.fa_proj = fa_proj
        self.audio_proj = audio_proj
        self.fusion_block = fusion_block
        self.final_classifier = final_classifier
        self._extracted_audio_feature = None

        for param in self.eye_track_model.parameters():
            param.requires_grad = False
        for param in self.noise_classifier.parameters():
            param.requires_grad = False
        for param in self.face_mesh_model.face_alignment_net.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNetV2 입력 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, cropped_faces, audio_features):
        device = next(self.parameters()).device
        
        cropped_faces = cropped_faces.to(device)
        audio_features = audio_features.to(device)
        
        # 배치 텐서인 경우 직접 처리
        if isinstance(cropped_faces, torch.Tensor):
            if cropped_faces.dim() == 4:  # [B, 3, 224, 224]
                batch_size = cropped_faces.shape[0]
                
                # MobileNet용 입력 (224x224로 리사이즈 필요)
                mobilenet_batch = F.interpolate(cropped_faces, size=(224, 224), mode='bilinear', align_corners=False)

                mean = torch.tensor([0.485, 0.456, 0.406], device=cropped_faces.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=cropped_faces.device).view(1, 3, 1, 1)
                
                # L2CS용 입력 
                mobilenet_batch = (mobilenet_batch - mean) / std
                l2cs_batch = (cropped_faces - mean) / std

                # 배치로 face landmarks 추출
                face_points_list = []
                for i in range(batch_size):
                    # 텐서를 numpy로 변환하여 face mesh 모델에 입력
                    face_img = cropped_faces[i].permute(1, 2, 0).cpu().numpy()
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype(np.uint8)
                    
                    # RGB → BGR 변환
                    face_crop_np = face_img[..., ::-1]
                    landmarks_list = self.face_mesh_model.get_landmarks(face_crop_np)
                    
                    if landmarks_list is None or len(landmarks_list) == 0:
                        face_points = np.zeros((68, 2), dtype=np.float32)
                    else:
                        face_points = landmarks_list[0]
                    
                    face_points_flat = face_points.flatten()  # [136]
                    face_points_list.append(face_points_flat)
                
                face_points_batch = torch.tensor(face_points_list, device=device, dtype=torch.float32)
                
            else:
                raise ValueError(f"Expected 4D tensor for batch processing, got {cropped_faces.dim()}D")

        img_features = self.mobile_net(mobilenet_batch)
        yaw_f, pitch_f, eye_features = self.eye_track_model(l2cs_batch)

        yaw, pitch = predict(yaw_f[i:i+1], pitch_f[i:i+1])
        
        mobilenet_features = self.mobilenet_proj(img_features)
        l2cs_features = self.l2cs_proj(eye_features)
        fa_features = self.fa_proj(face_points_batch)
        z_visual = torch.stack([mobilenet_features, l2cs_features, fa_features], dim=1)

        audio_feature, audio_class = self.noise_classifier(audio_features)
        audio_class = F.softmax(audio_class, dim=1)
        
        z_audio = self.audio_proj(audio_feature)

        fused_feature = self.fusion_block(z_visual, z_audio)
        class_output = self.final_classifier(fused_feature)

        return z_visual, z_audio, class_output, [yaw, pitch], audio_class

def load_model():
    # L2CS 객체 초기화
    l2cs = l2cs_load()
    # MediaPipe 객체 초기화
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu")
    # NoiseClassifier 객체 초기화
    noise_classifier = noise_classifier_load()
    # FaceBox 객체 초기화
    face_box = face_box_load()

    # Assume visual_proj and audio_proj output 256-dim each
    mobile_net_proj = ProjectionHead(input_dim=1280, proj_dim=256)
    l2cs_proj = ProjectionHead(input_dim=2048, proj_dim=256)
    fa_proj = ProjectionHead(input_dim=136, proj_dim=256)
    audio_proj = ProjectionHead(input_dim=256, proj_dim=256)

    fusion_block = TransformerFusion(embed_dim=256, num_heads=4, num_layers=3)
    final_classifier = FinalClassifier(embed_dim=256, num_classes=5)

    e2e_model = EndtoEndModel(
        mobile_net=MobileNetModel(),
        eye_track_model=l2cs,
        noise_classifier=noise_classifier,
        face_mesh_model=fa,
        mobilenet_proj=mobile_net_proj,
        l2cs_proj=l2cs_proj,
        fa_proj=fa_proj,
        audio_proj=audio_proj,
        fusion_block=fusion_block,
        final_classifier=final_classifier
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = torch.load(
        config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False
    )
    e2e_model.load_state_dict(results["ema"].state_dict(), strict=False)
    face_box.eval()
    e2e_model.eval()
    return face_box, e2e_model

def warmup_model(face_box, e2e_model, jpg_input_shape=(1,3,640,640), jpg2_input_shape=(1,3,448,448), aud_input_shape=(1,25)):
    "빠른 추론을 위한 warmup 기능"
    logger.info("Starting Warmup")
    face_box.eval()
    e2e_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_jpg_input = torch.rand(jpg_input_shape).to(device) * 255
    dummy_jpg2_input = torch.rand(jpg2_input_shape).to(device) * 255
    dummy_aud_input = torch.randn(aud_input_shape).to(device)
    with torch.no_grad():
        for _ in range(2):  # 2회 정도 실행
            face_box(dummy_jpg_input)
            result = e2e_model(dummy_jpg2_input, dummy_aud_input)
    logger.info("Warmup Finished")
    return result


def run(face_box, e2e_model, img, aud):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_box = face_box.to(device)
    e2e_model = e2e_model.to(device)
    aud = aud.to(device)

    def preprocess_batch_tensorized(face_box, img):
        """Batch-optimized face crop using GPU + Tensor only."""

        def resize_tensor_image(img_tensor, size=(640, 640)):
            return F.interpolate(
                img_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)

        def resize_cropped_face(face, size=(448, 448)):
            return F.interpolate(
                face.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)

        if isinstance(img, Image.Image):
            img_tensor = transforms.ToTensor()(img)
        elif isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = img

        img_tensor = resize_tensor_image(img_tensor)
        if torch.cuda.is_available() and next(face_box.parameters()).is_cuda:
            img_tensor = img_tensor.cuda(non_blocking=True)

        with torch.no_grad():
            try:
                result = face_box.predict(img_tensor.unsqueeze(0), verbose=False)
                result = result[0]

                try:
                    if len(result.boxes) > 0:
                        box = result.boxes[0].xyxy[0]
                        x1, y1, x2, y2 = box.int().tolist()
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = (
                            min(img_tensor.shape[-1], x2),
                            min(img_tensor.shape[-2], y2),
                        )
                        cropped = img_tensor[:, y1:y2, x1:x2]
                        if cropped.shape[-1] > 10 and cropped.shape[-2] > 10:
                            cropped_face = resize_cropped_face(cropped)
                            cropped_face = cropped_face.cpu()
                        else:
                            cropped_face = resize_cropped_face(img_tensor)
                            cropped_face = cropped_face.cpu()
                    else:
                        cropped_face = resize_cropped_face(img_tensor)
                        cropped_face = cropped_face.cpu()
                except Exception as e:
                    logger.error(f"Crop failed for image : {e}")
                    cropped_face = resize_cropped_face(img_tensor)  # ← 요기도
                    cropped_face = cropped_face.cpu()

            except Exception as e:
                logger.error(f"YOLO batch prediction error: {e}")
                cropped_face = img_tensor.cpu()

        # 배치 텐서로 반환 (일관성을 위해)
        if cropped_face is not None and cropped_face.numel() > 0:
            return cropped_face.unsqueeze(0)  # [B, 3, 224, 224]
        else:
            return torch.empty(0, 3, 448, 448)  # 빈 배치

    face_box.eval()
    e2e_model.eval()
    with torch.no_grad():
        cropped_face = preprocess_batch_tensorized(face_box, img)
        cropped_face = cropped_face.to(device)
        _, _, class_output, [yaw, pitch], audio_class = e2e_model(cropped_face, aud)
        predicted = torch.argmax(class_output, dim=1).item()
        noise_predicted = torch.argmax(audio_class, dim=1).item()
        label_dict = {
            0: "집중_흥미로움",
            1: "집중_차분함",
            2: "집중하지않음_차분함",
            3: "집중하지않음_지루함",
            4: "졸음_알수없음",
        }
        noise_label_dict = {
            0: "없음",
            1: "무음",
            2: "자동차",
            3: "싸이렌",
            4: "키보드",
            5: "대화소리",
        }
        result = label_dict.get(int(predicted), "Unknown")
        noise_result = noise_label_dict.get(int(noise_predicted), "Unknown")
        return (
            (int(predicted), result),
            (yaw, pitch),
            (int(noise_predicted), noise_result),
        )

if __name__=="__main__":
    face_box, e2e = load_model()
    result = warmup_model(face_box, e2e)
    print(result)