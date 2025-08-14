import logging
import torch
import torchaudio
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FastAudioPreprocessor:
    def __init__(self, scaler, sr=48000, frame_size=512, hop_size=256, device="cuda"):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=8, melkwargs={"n_fft": 512, "hop_length": 256}
        ).to(self.device)

        logger.info(f"🎧 Using device: {self.device}")

    def preprocess_wav(self, wav_file):
        """n_jobs=1로 GPU 메모리 충돌 방지"""
        logger.info("🎵 Starting audio processing...")
        # GPU 사용시 순차 처리, CPU 사용시 병렬 처리
        try:
            features_tensor = self._process_single_file(wav_file)
        except Exception as e:
            logger.error(f"Error processing : {e}")
        return features_tensor

    def _process_single_file(self, wav_file):
        try:
            waveform, orig_sr = librosa.load(wav_file, sr=48000)
            waveform = torch.tensor(waveform, device=self.device).unsqueeze(
                0
            )  # [1, samples] 형태로

            if orig_sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sr).to(
                    self.device
                )
                waveform = resampler(waveform.to(self.device))
            else:
                waveform = waveform.to(self.device)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)  # [samples]

            if len(waveform) < self.frame_size:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.frame_size - len(waveform))
                )

            frames = self.frame_signal(waveform)

            # 🚀 2D 보장 - 조건문 제거로 속도 향상
            if frames.shape[0] == 0:
                features = torch.zeros(25, device=self.device)
            else:
                if frames.ndim == 1:
                    frames = frames.unsqueeze(0)  # [1, frame_size]

                # 이제 무조건 2D 입력
                frame_features = self.extract_features(frames)  # [num_frames, 21]
                features = self._compute_statistical_features(frame_features)

            if self.scaler is not None:
                features_np = features.cpu().numpy().reshape(1, -1)
                features_np = self.scaler.transform(features_np).flatten()
                features = torch.tensor(
                    features_np, device=self.device, dtype=torch.float32
                )

            return features

        except Exception as e:
            logger.error(f"Error processing : {e}")
            return None

    def frame_signal(self, signal):
        """벡터화된 GPU 기반 프레이밍 (stride trick)"""
        num_frames = 1 + (len(signal) - self.frame_size) // self.hop_size
        if num_frames <= 0:
            return torch.empty(0, self.frame_size, device=self.device)

        # Window
        window = torch.hamming_window(self.frame_size, device=self.device)

        # Framing via unfold (faster than loop)
        frames = signal.unfold(
            0, self.frame_size, self.hop_size
        )  # [num_frames, frame_size]
        frames = frames * window  # Apply window

        # Silence frame 제거 (optional)
        valid = torch.sum(torch.abs(frames), dim=1) > 1e-8
        return frames[valid]

    def compute_power_spectrum(self, frame):
        """GPU 기반 파워 스펙트럼"""
        spectrum = torch.fft.rfft(frame)
        power_spectrum = torch.abs(spectrum) ** 2
        return torch.mean(power_spectrum)

    def compute_cepstrum(self, frame, num_coeffs=13):
        """GPU 기반 켑스트럼 - 원본과 동일하게 13개 계산 후 선택"""
        spectrum = torch.fft.rfft(frame)
        log_spectrum = torch.log(torch.abs(spectrum) + 1e-10)
        cepstrum = torch.real(torch.fft.ifft(log_spectrum))
        return cepstrum[:num_coeffs]  # 13개 반환

    def compute_spectral_features(self, frame):
        """GPU 기반 스펙트럴 특징"""
        spectrum = torch.abs(torch.fft.rfft(frame))
        freqs = torch.fft.fftfreq(len(frame), d=1 / self.sr, device=self.device)
        freqs = freqs[: len(freqs) // 2]
        spectrum = spectrum[: len(spectrum) // 2]

        # Spectral centroid
        centroid = torch.sum(freqs * spectrum) / (torch.sum(spectrum) + 1e-10)

        # Spectral rolloff
        cumsum_spectrum = torch.cumsum(spectrum, dim=0)
        rolloff_idx = torch.where(cumsum_spectrum >= 0.85 * cumsum_spectrum[-1])[0]
        rolloff = (
            freqs[rolloff_idx[0]]
            if len(rolloff_idx) > 0
            else torch.tensor(0.0, device=self.device)
        )

        # Spectral bandwidth
        bandwidth = torch.sqrt(
            torch.sum(((freqs - centroid) ** 2) * spectrum)
            / (torch.sum(spectrum) + 1e-10)
        )

        # 모든 값이 스칼라인지 확인하고 텐서로 변환
        centroid = centroid if centroid.dim() == 0 else centroid.squeeze()
        rolloff = rolloff if rolloff.dim() == 0 else rolloff.squeeze()
        bandwidth = bandwidth if bandwidth.dim() == 0 else bandwidth.squeeze()

        return torch.tensor([centroid, rolloff, bandwidth], device=self.device)

    def compute_mfcc(self, frame):
        """GPU 기반 MFCC"""
        try:
            # [1, 1, samples] 형태로 변환 (배치, 채널, 샘플)
            frame_2d = frame.unsqueeze(0).unsqueeze(0)
            mfcc = self.mfcc_transform(frame_2d)  # [1, n_mfcc, time_frames]

            # 시간축 평균
            mfcc = torch.mean(mfcc, dim=2).squeeze(0)  # [n_mfcc]

            assert mfcc.dim() == 1, f"MFCC shape mismatch: {mfcc.shape}"
            return mfcc
        except:
            return torch.zeros(8, device=self.device)

    def extract_features(self, frames):
        """2D 전용 고속 특징 추출 - frames: [B, frame_size]"""
        B = frames.shape[0]

        # FFT 계산 (한 번만)
        spectrum = torch.fft.rfft(frames, dim=1)
        spectrum_abs = torch.abs(spectrum)
        log_spectrum = torch.log(spectrum_abs + 1e-10)

        # 1. Cepstrum [B, 8]
        cep = torch.real(torch.fft.irfft(log_spectrum, dim=1))[:, :8]

        # 2. Power spectrum [B, 1]
        power = torch.mean(spectrum_abs**2, dim=1, keepdim=True)

        # 3. Spectral features [B, 3]
        freqs = torch.fft.rfftfreq(self.frame_size, d=1 / self.sr, device=self.device)
        total_energy = torch.sum(spectrum_abs, dim=1, keepdim=True) + 1e-10

        # Centroid
        centroid = torch.sum(freqs * spectrum_abs, dim=1) / total_energy.squeeze(1)

        # Rolloff (벡터화)
        cumsum_spec = torch.cumsum(spectrum_abs, dim=1)
        threshold = 0.85 * cumsum_spec[:, -1:]
        rolloff_idx = torch.argmax((cumsum_spec >= threshold).float(), dim=1)
        rolloff_idx = torch.clamp(rolloff_idx, 0, len(freqs) - 1)
        rolloff = freqs[rolloff_idx]

        # Bandwidth
        bandwidth = torch.sqrt(
            torch.sum(((freqs - centroid.unsqueeze(1)) ** 2) * spectrum_abs, dim=1)
            / total_energy.squeeze(1)
        )

        spectral_features = torch.stack([centroid, rolloff, bandwidth], dim=1)

        # 4. MFCC [B, 8] - 수정된 부분
        mfcc_features = []
        for i in range(B):
            # 각 프레임을 개별적으로 처리: [1, 1, frame_size]
            frame_single = frames[i]
            frame_for_mfcc = frame_single.unsqueeze(0).unsqueeze(0)
            mfcc_single = self.mfcc_transform(
                frame_for_mfcc
            )  # [1, n_mfcc, time_frames]
            mfcc_single = mfcc_single.squeeze(0).squeeze(0)
            mfcc_single = torch.mean(mfcc_single, dim=1)  # [8] (평균을 시간축에 대해)
            mfcc_features.append(mfcc_single)

        mfcc = torch.stack(mfcc_features, dim=0)  # [B, 8]

        # 5. ZCR [B, 1]
        zcr = torch.mean(
            (frames[:, 1:] * frames[:, :-1] < 0).float(), dim=1, keepdim=True
        )

        # 한 번에 concat [B, 21]
        return torch.cat([cep, power, spectral_features, mfcc, zcr], dim=1)

    def _compute_statistical_features(self, features):
        """GPU 기반 통계 특징 계산"""
        # features: [num_frames, 21]
        feature_mean = torch.mean(features, dim=0)  # [21]
        feature_std = torch.std(features, dim=0)  # [21]
        key_std = feature_std[:4]  # 첫 4개 std만 선택

        # [21 + 4 = 25] 특징 반환
        return torch.cat([feature_mean, key_std])


def load_preprocessor(scaler_path):
    load_path = scaler_path
    scaler = joblib.load(load_path)
    preprocessor = FastAudioPreprocessor(scaler=scaler)
    return preprocessor


# 사용법 함수
def preprocess_audio_data(preprocessor, wav_file):
    """오디오 데이터셋 전처리 실행"""
    feature_tensor = preprocessor.preprocess_wav(wav_file)
    return feature_tensor


if __name__ == "__main__":
    file_path = "C:\\Users\\82102\\multimodal\\temp\\combined_000_8b417667-17a0-4b95-b588-6cf783b22855.wav"
    preprocessor = load_preprocessor("models/train_dataset_scaler_gpu.pkl")
    preprocess_audio_data(preprocessor, file_path)
