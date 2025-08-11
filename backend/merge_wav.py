import struct
import logging

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def merge_wav_chunks(wav_chunks_bytes):
    """
    여러 개의 WAV 파일 바이트를 합쳐서 하나의 WAV 파일로 만들기
    
    Args:
        wav_chunks_bytes (list): WAV 파일들의 바이트 데이터 리스트
    
    Returns:
        bytes: 합쳐진 WAV 파일의 바이트 데이터
    """
    if not wav_chunks_bytes:
        return b''
    
    # 첫 번째 WAV에서 헤더 정보 추출
    first_wav = wav_chunks_bytes[0]
    if len(first_wav) < 44:
        raise ValueError("유효하지 않은 WAV 파일")
    
    # WAV 헤더 파싱
    header = first_wav[:44]
    
    # 헤더에서 오디오 포맷 정보 추출
    num_channels = struct.unpack('<H', header[22:24])[0]
    sample_rate = struct.unpack('<I', header[24:28])[0]
    bits_per_sample = struct.unpack('<H', header[34:36])[0]
    byte_rate = struct.unpack('<I', header[28:32])[0]
    block_align = struct.unpack('<H', header[32:34])[0]
    
    logger.info(f"WAV 정보: {num_channels}채널, {sample_rate}Hz, {bits_per_sample}bit")
    
    # 모든 WAV 파일의 오디오 데이터 부분만 추출해서 합치기
    combined_audio_data = b''
    
    for i, wav_bytes in enumerate(wav_chunks_bytes):
        if len(wav_bytes) < 44:
            logger.error(f"WARNING: 청크 {i}가 너무 작음, 건너뜀")
            continue
            
        # WAV 헤더(44바이트) 건너뛰고 오디오 데이터만 추출
        audio_data = wav_bytes[44:]
        combined_audio_data += audio_data
    
    # 새로운 WAV 헤더 생성
    total_file_size = 44 + len(combined_audio_data)
    audio_data_size = len(combined_audio_data)
    
    # WAV 헤더 구성
    new_header = bytearray(44)
    
    # RIFF 헤더
    new_header[0:4] = b'RIFF'
    struct.pack_into('<I', new_header, 4, total_file_size - 8)  # 파일크기 - 8
    new_header[8:12] = b'WAVE'
    
    # fmt 청크
    new_header[12:16] = b'fmt '
    struct.pack_into('<I', new_header, 16, 16)  # fmt 청크 크기
    struct.pack_into('<H', new_header, 20, 1)   # PCM 포맷
    struct.pack_into('<H', new_header, 22, num_channels)
    struct.pack_into('<I', new_header, 24, sample_rate)
    struct.pack_into('<I', new_header, 28, byte_rate)
    struct.pack_into('<H', new_header, 32, block_align)
    struct.pack_into('<H', new_header, 34, bits_per_sample)
    
    # data 청크
    new_header[36:40] = b'data'
    struct.pack_into('<I', new_header, 40, audio_data_size)
    
    # 최종 WAV 파일 = 새 헤더 + 합쳐진 오디오 데이터
    final_wav = bytes(new_header) + combined_audio_data
    
    # 길이 계산 (대략적)
    duration = len(combined_audio_data) / (sample_rate * num_channels * (bits_per_sample // 8))
    
    return final_wav


# 실제 사용할 함수들
def split_combined_wav_chunks(combined_wav_bytes):
    """
    Python에서 받은 합쳐진 WAV 바이트들을 개별 청크로 분리
    (만약 여러 WAV가 연결된 상태로 온다면)
    """
    chunks = []
    offset = 0
    
    while offset < len(combined_wav_bytes):
        # RIFF 헤더 찾기
        if offset + 8 > len(combined_wav_bytes) or combined_wav_bytes[offset:offset+4] != b'RIFF':
            # RIFF 헤더가 없으면 전체를 하나의 WAV로 처리
            if not chunks and len(combined_wav_bytes) > 44:
                return [combined_wav_bytes]
            break
            
        # 이 청크의 크기 읽기
        try:
            chunk_size = struct.unpack('<I', combined_wav_bytes[offset+4:offset+8])[0]
            total_chunk_size = chunk_size + 8  # RIFF + 크기 필드 포함
            
            # 경계 확인
            if offset + total_chunk_size > len(combined_wav_bytes):
                chunk = combined_wav_bytes[offset:]
                chunks.append(chunk)
                break
            
            # 청크 추출
            chunk = combined_wav_bytes[offset:offset + total_chunk_size]
            chunks.append(chunk)
            
            offset += total_chunk_size
            
        except struct.error as e:
            logger.error(f"청크 파싱 오류: {e}")
            break
    
    return chunks


# SessionAudioBuffer에서 사용할 수 있도록 수정된 버전
def merge_wav_chunks_from_buffer(buffer_bytes):
    """
    SessionAudioBuffer의 current_buffer에서 여러 WAV를 합치기
    """
    # 버퍼에서 개별 WAV 청크들 분리
    wav_chunks = split_combined_wav_chunks(buffer_bytes)
    
    if not wav_chunks:
        print("WAV 청크를 찾을 수 없음")
        return buffer_bytes  # 원본 반환
    
    print(f"{len(wav_chunks)}개의 WAV 청크 발견")
    
    # 청크들을 합치기
    merged_wav = merge_wav_chunks(wav_chunks)
    
    return merged_wav