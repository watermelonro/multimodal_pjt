# 멀티모달 집중도 분석 및 피드백 시스템

이 프로젝트는 사용자의 웹캠 영상과 마이크 입력을 실시간으로 분석하여, 학습 집중도를 측정하고 개인화된 피드백을 제공하는 시스템입니다.

## 주요 기능

- **실시간 집중도 분석**: 웹캠 영상의 얼굴 방향, 눈 깜빡임, 시선 등을 분석하여 실시간으로 집중/비집중 상태를 추론합니다.
- **소음 분석**: 마이크 입력으로부터 주변 소음을 분류합니다.
- **LLM 기반 피드백**: 세션 종료 시, 수집된 데이터를 바탕으로 Upstage Solar LLM이 학습 태도와 내용에 대한 종합적인 피드백 리포트를 생성합니다.
- **시각화**: 분석된 집중도 흐름을 그래프로 시각화하여 제공합니다.
- **웹 인터페이스**: React로 구축된 프론트엔드를 통해 사용자와 상호작용합니다.

## 기술 스택

- **Backend**: Python, FastAPI, PyTorch, LangChain, Upstage API
- **Frontend**: JavaScript, React, Tailwind CSS
- **Model**: ResNet, MobileNetV2, YOLO(ultralytics), L2CS, Transformer-based fusion model

## 프로젝트 구조

```
.
├── backend/         # FastAPI 백엔드 서버
│   ├── app.py       # (사용되지 않음, main.py가 주 진입점)
│   ├── config.py    # 설정 변수
│   ├── data_process.py # 수집 데이터 후처리
│   ├── feedback_generator.py # 피드백 생성 클래스
│   ├── graph.py     # 집중도 그래프 생성
│   ├── llm_prompt.py # LLM 프롬프트 및 파이프라인
│   ├── main.py      # API 및 웹소켓 서버 메인
│   ├── model_inference.py # 모델 추론 로직
│   └── ...
├── frontend/        # React 프론트엔드
│   ├── src/App.jsx  # 메인 애플리케이션 컴포넌트
│   └── ...
├── models/          # 모델 가중치 파일 (.pth, .pkl)
├── data/            # 데이터 (PDF, 벡터 스토어 등)
└── requirements.txt # Python 라이브러리 종속성
└── .env # API 키 등 저장
```

## 설치 및 실행 방법

### 사전 준비

- Python 3.10 이상
- Node.js 및 npm
- `.env` 파일 설정: `backend` 폴더와 같은 위치에 `.env` 파일을 생성하고 아래 내용을 추가해야 함.

  ```
  UPSTAGE_API_KEY="여기에_업스테이지_API_키를_입력하세요"
  ```

### 1. 백엔드 설정 및 실행

1.  **가상환경 생성 및 활성화 (권장)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

2.  **필요한 라이브러리 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **백엔드 서버 실행**:
    `multimodal_pjt` 폴더를 기준으로 아래 명령어를 실행합니다.
    ```bash
    python backend/main.py
    ```
    서버가 `http://localhost:8000`에서 실행됩니다.

### 2. 프론트엔드 설정 및 실행

1.  **프론트엔드 폴더로 이동**:
    ```bash
    cd frontend
    ```

2.  **필요한 패키지 설치**:
    ```bash
    npm install
    ```

3.  **프론트엔드 개발 서버 실행**:
    ```bash
    npm run dev
    ```
    웹 애플리케이션이 `http://localhost:5173`에서 열립니다.
