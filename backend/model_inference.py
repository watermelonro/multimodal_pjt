import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from eye_tracker.L2CS import load_model as l2cs_load
from eye_tracker.L2CS import predict
from noise_classifier_ast.classifier import load_model as noise_classifier_load
from face_box.yolo import load_model as face_box_load
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import face_alignment
import warnings
import logging
from wav_process import FastAudioPreprocessor, preprocess_audio_data
import cv2
import matplotlib.pyplot as plt
import os
import math

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

audio_conf = {
    'num_mel_bins': 128, 
    'target_length': 1024, 
    'freqm': 48, 
    'timem': 192,  
    'dataset': 'aihub_audio_dataset', 
    'mean':-4.2677393, 
    'std':4.5689974
    }

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

    def forward(self, cropped_faces, audio_features):
        device = next(self.parameters()).device
        
        cropped_faces = cropped_faces.to(device)
        audio_features = audio_features.to(device)

        # 검증을 위한 값
        global_face_points = []
        
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
                    global_face_points.append(face_points)
                    
                    face_points_flat = face_points.flatten()  # [136]
                    face_points_list.append(face_points_flat)
                
                face_points_batch = torch.tensor(face_points_list, device=device, dtype=torch.float32)
                
            else:
                raise ValueError(f"Expected 4D tensor for batch processing, got {cropped_faces.dim()}D")

        img_features = self.mobile_net(mobilenet_batch)
        with torch.no_grad():
            yaw_f, pitch_f, eye_features = self.eye_track_model(l2cs_batch)
            yaw, pitch = predict(yaw_f[i:i+1], pitch_f[i:i+1])
        
        mobilenet_features = self.mobilenet_proj(img_features)
        l2cs_features = self.l2cs_proj(eye_features)
        fa_features = self.fa_proj(face_points_batch)
        z_visual = torch.stack([mobilenet_features, l2cs_features, fa_features], dim=1)

        with torch.no_grad():
            audio_class_logits, audio_feature = self.noise_classifier(audio_features)
            audio_class = torch.sigmoid(audio_class_logits)
        
        z_audio = self.audio_proj(audio_feature)

        fused_feature = self.fusion_block(z_visual, z_audio)
        class_output = self.final_classifier(fused_feature)

        return z_visual, z_audio, class_output, (yaw, pitch), audio_class, (cropped_faces[0].cpu(), global_face_points[0]), audio_features[0].cpu()

def load_model():
    # L2CS 객체 초기화
    l2cs = l2cs_load()
    # MediaPipe 객체 초기화
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
    # NoiseClassifier 객체 초기화
    noise_classifier = noise_classifier_load()
    # FaceBox 객체 초기화
    face_box = face_box_load()

    # Assume visual_proj and audio_proj output 256-dim each
    mobile_net_proj = ProjectionHead(input_dim=1280, proj_dim=256)
    l2cs_proj = ProjectionHead(input_dim=2048, proj_dim=256)
    fa_proj = ProjectionHead(input_dim=136, proj_dim=256)
    audio_proj = ProjectionHead(input_dim=768, proj_dim=256)

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
    
    # Determine the device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    results = torch.load(
            config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False
        )
    e2e_model.load_state_dict(results, strict=False)
    e2e_model.eye_track_model = l2cs
    e2e_model.noise_classifier = noise_classifier
    e2e_model.face_mesh_model = fa
    
    face_box.eval()
    e2e_model.eval()
    return face_box, e2e_model

def warmup_model(face_box, e2e_model, check=True):
    "빠른 추론을 위한 warmup 기능"
    try:
        logger.info("Starting Warmup")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        face_box = face_box.eval().to(device)
        e2e_model = e2e_model.eval().to(device)
        pad = FastAudioPreprocessor(audio_conf)
        img = Image.open(os.path.join(config.WARMUP_PATH, 'temp_image.jpg')).convert("RGB")
        aud = preprocess_audio_data(pad, os.path.join(config.WARMUP_PATH, 'temp_audio.wav'))

        with torch.no_grad():
            for _ in range(2):  # 2회 정도 실행
                result, audio_inputs = run(face_box, e2e_model, img, aud, check)
        logger.info("Warmup Finished")
    except Exception as e:
        logger.error(f"Warmup 중 오류 발생: {e}")
        return None, None
    return result, audio_inputs


def run(face_box, e2e_model, img, aud, check=False):
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
                    cropped_face = resize_cropped_face(img_tensor)
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
        _, _, class_output,(yaw, pitch), audio_class, visual_outputs, audio_inputs = e2e_model(cropped_face, aud)
        predicted = torch.argmax(class_output, dim=1).item()
        noise_predicted = torch.argmax(audio_class[..., :-1], dim=1).item()
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
        if check:
            check_visual(visual_outputs[0], visual_outputs[1], (yaw, pitch), result, audio_inputs, noise_result)
        return (
            (int(predicted), result),
            (yaw, pitch),
            (int(noise_predicted), noise_result)
        ), audio_inputs

def check_visual(cropped_faces, face_points, pose, result, audio_inputs, noise_result):
    """
    눈 처리 과정에서 오류 발생 시 디버깅용 시각화
    
    Args:
        cropped_faces: torch.Tensor [3, 448, 448] 
        face_points: numpy array [68, 2] - face landmark 좌표
        pose: yaw, pitch 시선 각도
        result: str 형식 예측결과
        audio_inputs: 입력된 오디오 특성
    """
    
    # 1. 원본 얼굴 이미지 추출
    face_img = cropped_faces.permute(1, 2, 0).numpy()
    
    if face_img.max() <= 1.0:
        face_img = (face_img * 255).astype(np.uint8)
    
    # 2. 시각화 준비
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'Face Check', fontsize=16)
    
    # 2-1. 원본 얼굴 이미지 (RGB)
    axes[0, 0].imshow(face_img)
    axes[0, 0].set_title('Original Face (RGB)')
    axes[0, 0].axis('off')
    
    # 2-2. Face landmarks 표시
    face_with_landmarks = face_img.copy()
    
    # 모든 landmark 점 표시
    for i, (x, y) in enumerate(face_points):
        cv2.circle(face_with_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)
        if i % 10 == 0:  # 10개마다 번호 표시
            cv2.putText(face_with_landmarks, str(i), (int(x)+3, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 눈 landmark 강조 (36-47)
    left_eye_points = face_points[36:42]  # 왼쪽 눈
    right_eye_points = face_points[42:48]  # 오른쪽 눈
    
    for (x, y) in left_eye_points:
        cv2.circle(face_with_landmarks, (int(x), int(y)), 3, (255, 0, 0), -1)  # 빨간색
    for (x, y) in right_eye_points:
        cv2.circle(face_with_landmarks, (int(x), int(y)), 3, (0, 0, 255), -1)  # 파란색
    
    axes[0, 1].imshow(face_with_landmarks)
    axes[0, 1].set_title('Face with Landmarks\n(Red: Left Eye, Blue: Right Eye)')
    axes[0, 1].axis('off')
    
    # 2-3. 좌표 정보 텍스트
    coord_text = f"Left Eye Points (36-41):\n"
    for i, (x, y) in enumerate(left_eye_points):
        coord_text += f"  {36+i}: ({x:.1f}, {y:.1f})\n"
    
    coord_text += f"\nRight Eye Points (42-47):\n"
    for i, (x, y) in enumerate(right_eye_points):
        coord_text += f"  {42+i}: ({x:.1f}, {y:.1f})\n"
    
    axes[0, 2].text(0.5, 0.5, coord_text, transform=axes[0, 2].transAxes,
                    fontsize=10, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
    axes[0, 2].set_title('Eye Coordinates')
    axes[0, 2].axis('off')

    # 시선처리 이미지
    yaw, pitch = pose[0], pose[1]
    h, w = face_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    length = 100
    dx = -length * math.sin(math.radians(yaw))
    dy = -length * math.sin(math.radians(pitch))
    axes[1, 0].imshow(face_img)
    axes[1, 0].arrow(center_x, center_y, dx, dy, head_width=10, head_length=15, fc='red', ec='red')
    axes[1, 0].plot(center_x, center_y, 'bo')
    axes[1, 0].set_title('Pose')
    axes[1, 0].axis('off')
    pose_info = f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°"
    axes[1, 1].text(0.5, 0.5, pose_info, transform=axes[1, 1].transAxes,
                    fontsize=14, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))
    axes[1, 1].axis('off')

    # 최종 예측 결과
    result_info = f"Final Prediction : {result}"
    axes[1, 2].text(0.5, 0.5, result_info, transform=axes[1, 2].transAxes,
                    fontsize=14, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 2].axis('off')
    
    # 오디오 특성
    axes[2, 0].remove()
    axes[2, 1].remove() 
    ax_combined = plt.subplot2grid((3, 3), (2, 0), colspan=2, fig=fig)
    im = ax_combined.imshow(audio_inputs.T, aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax_combined, format="%+2.0f dB")
    ax_combined.set_title("Mel Filter Bank Features (fbank)")
    ax_combined.set_xlabel("Time frames")
    ax_combined.set_ylabel("Mel bins")

    # 잡음 예측 결과 (1칸)
    audio_info = f"Noise Prediction:\n{noise_result}"
    axes[2, 2].text(0.5, 0.5, audio_info, transform=axes[2, 2].transAxes,
                fontsize=14, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    axes[2, 2].set_title('Audio Result')
    axes[2, 2].axis('off')

    plt.tight_layout()    
    plt.savefig(os.path.join(config.WARMUP_PATH, 'check.png'), dpi=150, bbox_inches='tight')
    plt.close()
