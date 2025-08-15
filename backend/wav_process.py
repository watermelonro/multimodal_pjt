import logging
import torch
import torchaudio
import librosa
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FastAudioPreprocessor:
    def __init__(self, audio_conf, sr=16000):
        self.sr = sr
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        
    def _wav2fbank(self, filename):
        target_sr = 16000

        def resample_if_necessary(waveform, orig_sr):
            if orig_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                waveform = resampler(waveform)
            return waveform
        
        waveform, sr = torchaudio.load(filename)
        waveform = resample_if_necessary(waveform, sr)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def preprocess_wav(self, data_path):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        fbank = self._wav2fbank(data_path)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        # the output fbank shape is [batch_size, time_frame_num, frequency_bins], e.g., [1, 1024, 128]
        return fbank.unsqueeze(0)

# 사용법 함수
def preprocess_audio_data(preprocessor, wav_file):
    """오디오 데이터셋 전처리 실행"""
    feature_tensor = preprocessor.preprocess_wav(wav_file)
    return feature_tensor

if __name__=="__main__":
    audio_conf = {
        'num_mel_bins': 128, 
        'target_length': 1024, 
        'freqm': 48, 
        'timem': 192,  
        'dataset': 'aihub_audio_dataset', 
        'mean':-4.2677393, 
        'std':4.5689974
        }
    file_path = "C:\\Users\\82102\\multimodal\\temp\\combined_000_8b417667-17a0-4b95-b588-6cf783b22855.wav"
    preprocessor = FastAudioPreprocessor(audio_conf)
    f_tensor = preprocess_audio_data(preprocessor, file_path)
    print(f_tensor.shape)