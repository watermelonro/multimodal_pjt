import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# --- API Keys ---
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    print("Warning: UPSTAGE_API_KEY is not set in the .env file.")

# --- Paths ---
# 프로젝트의 루트 디렉토리
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 정적 파일(생성된 그래프 등)을 저장할 디렉토리
# main.py에서 StaticFiles를 마운트할 때 사용
STATIC_DIR = os.path.join(
    PROJECT_ROOT, "backend", "static"
)  # 백엔드 폴더 내에 static 생성

# 데이터 및 모델 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ChromaDB 벡터 스토어 경로
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_store", "경영정보시스템5_chroma_db")

# 모델 체크포인트 경로
MODEL_CHECKPOINT_PATH = os.path.join(
    MODELS_DIR, "best.pt"
)

TEMP_DIR_PATH = os.path.join(PROJECT_ROOT, "temp")

# --- Analysis Parameters ---
# 클라이언트로부터 수신하는 비디오의 초당 프레임 수 (가정)
VIDEO_FPS = 15
TIMESTEP = 10.0

# --- Model Settings ---
# Solar LLM 모델 이름
LLM_MODEL_NAME = "solar-pro2"

# 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "embedding-query"

# 정적 디렉토리가 없으면 생성
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
