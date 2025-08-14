import asyncio
import json
from datetime import datetime
import io
import base64
import uuid
import logging
import os
from typing import Dict, Any
from queue import Queue
import queue
import threading
import time

import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pymongo import MongoClient

# --- 프로젝트 모듈 임포트 ---
import config
import model_inference
from feedback_generator import GenerateFeedback
from data_process import analyze_concentration_changes
from wav_process import load_preprocessor, preprocess_audio_data
from merge_wav import merge_wav_chunks_from_buffer as merge

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- CORS 미들웨어 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 생성된 그래프 이미지 등을 제공하기 위한 정적 파일 경로 마운트
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

client = MongoClient("mongodb://localhost:27017")
db = client["mydatabase"]
collection = db["sessions"]


def save_result(sessionid, result):
    collection.update_one(
        {"session_id": sessionid}, {"$push": {"results": result}}, upsert=True
    )


# 전역 세션 관리
session_buffers = {}
executor = ThreadPoolExecutor(max_workers=4)  # 동시 처리 가능한 세션 수


# --- 세션 관리를 위한 클래스 ---
class SessionManager:
    """웹소켓 클라이언트별 세션을 관리"""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_name: str, topic: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_name": user_name,
            "topic": topic,
            "start_time": datetime.now(),
        }
        logger.info(f"세션 생성됨: {session_id} (사용자: {user_name})")
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"세션 종료됨: {session_id}")


session_manager = SessionManager()


# --- 핵심 분석기 클래스 ---
class LectureAnalyzer:
    """모든 모델을 총괄하고 데이터 분석 파이프라인을 실행"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"분석기 초기화 중... (Device: {self.device})")
        try:
            self.pad = load_preprocessor(
                os.path.join(config.MODELS_DIR, "train_dataset_scaler_gpu.pkl")
            )
            logger.info("✅ 음성 전처리 모듈 로드 완료")
            # End-to-End 멀티모달 모델 로드
            self.face_box_model, self.e2e_model = model_inference.load_model()
            self.e2e_model.to(self.device)
            model_inference.warmup_model(self.face_box_model, self.e2e_model)
            logger.info("✅ 멀티모달 추론 모델 로드 완료")

        except Exception as e:
            logger.critical(f"❌ 모델 로딩 실패: {e}", exc_info=True)
            raise RuntimeError(
                "필수 모델 로딩에 실패하여 서버를 시작할 수 없습니다."
            ) from e

    def process_chunk(
        self, frame_data: bytes, audio_path: str, last_timestamp: float
    ) -> Dict | None:
        """실시간 데이터 청크를 받아 멀티모달 모델로 분석"""
        try:
            # 1. 데이터 전처리
            pil_image = Image.open(io.BytesIO(frame_data))

            pil_image.save("output.jpg")

            audio_tensor = preprocess_audio_data(self.pad, audio_path)

            # 2. 모델 추론 실행
            (pred_num, pred_str), (yaw, pitch), (noise_num, noise_str) = (
                model_inference.run(
                    self.face_box_model, self.e2e_model, pil_image, audio_tensor
                )
            )

            # 3. 결과 구조화
            start_time = last_timestamp - config.TIMESTEP
            result = {
                "timestamp": {"start": start_time, "end": last_timestamp},
                "result": {"num": pred_num, "str": pred_str},
                "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                "noise": {"num": noise_num, "str": noise_str},
                "text": f"({self._format_time(start_time)}~ {self._format_time(last_timestamp)}지점의 강의 내용) ",  # Whisper 연동 시 실제 텍스트로 대체
            }
            return result

        except Exception as e:
            logger.error(f"실시간 처리 중 오류 발생: {e}", exc_info=True)
            return None

    def generate_final_report(
        self, session_data: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """세션 종료 시 종합 리포트 생성"""
        user_name = session_data["user_name"]
        topic = session_data["topic"]

        logger.info(f"'{user_name}'님의 최종 리포트 생성 시작...")
        try:
            # 1. 피드백 생성 클래스 사용 (그래프 이미지가 HTML에 포함됨)
            feedback_generator = GenerateFeedback()
            doc = collection.find_one(
                {"session_id": session_id}, {"_id": 0, "results": 1}
            )
            results = doc.get("results", []) if doc else []
            if not results:
                logger.warning("분석할 데이터가 없어 리포트를 생성할 수 없습니다.")
                return {"error": "분석 데이터 부족"}
            full_html_report = feedback_generator.generate(
                topic=topic, name=user_name, data=results
            )
            logger.info("✅ LLM 리포트 생성 완료")

            # 2. 생성된 HTML 리포트 파일로 저장
            safe_user_name = "".join(c for c in user_name if c.isalnum())
            report_filename = f"feedback_{safe_user_name}_{uuid.uuid4().hex[:6]}.html"
            report_path_abs = os.path.join(config.STATIC_DIR, report_filename)

            try:
                with open(report_path_abs, "w", encoding="utf-8") as f:
                    f.write(full_html_report)
                logger.info(f"✅ HTML 리포트 파일 저장 완료: {report_path_abs}")
            except Exception as e:
                logger.error(f"HTML 리포트 파일 저장 중 오류 발생: {e}")

            # 3. 심층 분석 데이터 (선택 사항이지만 유지)
            sorted_results = sorted(results, key=lambda x: x["timestamp"]["start"])
            insights = analyze_concentration_changes(sorted_results)
            logger.info("✅ 데이터 심층 분석 완료")

            return {
                "user_name": user_name,
                "topic": topic,
                "llm_report": full_html_report,  # 전체 HTML을 전달
                "detailed_analysis": insights,
            }

        except Exception as e:
            logger.error(f"최종 리포트 생성 중 오류 발생: {e}", exc_info=True)
            return {"error": "리포트 생성 중 서버 오류 발생"}

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}분 {secs:02d}초"


# --- 전역 분석기 인스턴스 생성 ---
try:
    analyzer = LectureAnalyzer()
except RuntimeError as e:
    logger.critical(f"분석기 인스턴스 생성 실패. 서버를 종료합니다. 오류: {e}")
    analyzer = None


class SessionAudioBuffer:
    def __init__(self, session_id: str, analyzer):
        # 유저 정보
        self.session_id = session_id

        # 상태 확인용 (스레드 안전)
        self.num_chunks = 0
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._shutdown = False

        # 모델
        self.analyzer = analyzer

        # 데이터 저장
        self.buffer = b""
        self.frame_latest = None
        self.data_queue = Queue(maxsize=5)  # 백프레셔 방지

        # 모델 처리 스레드
        self.model_queue = Queue()
        self.model_thread = threading.Thread(target=self._model_worker)
        self.model_thread.start()

    def is_processing(self):
        with self._processing_lock:
            return self._is_processing

    def _set_processing(self, value):
        with self._processing_lock:
            self._is_processing = value

    def _model_worker(self):
        """모델 추론 전용 워커 스레드"""
        while not self._shutdown:
            try:
                task = self.model_queue.get_nowait()
                if task is None:  # 종료 신호
                    break

                start_time = time.time()
                logger.info(f"Session {task['session_id']}: 모델 추론 시작")

                # WAV 파일 저장
                wav_path = self._save_wav_file(task)

                try:
                    # 모델 추론
                    result = self.analyzer.process_chunk(
                        task["frame"], wav_path, task["timestamp"]
                    )

                    processing_time = time.time() - start_time
                    logger.info(
                        f"Session {task['session_id']}: 모델 추론 완료 ({processing_time:.2f}초)"
                    )

                    # MongoDB에 저장
                    save_result(self.session_id, result)

                except Exception as e:
                    logger.error(f"모델 추론 오류 (Session {task['session_id']}): {e}")
                finally:
                    # 파일 정리
                    self._cleanup_wav_file(wav_path)

            except queue.Empty:
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"모델 워커 오류: {e}")
            finally:
                # 처리 완료 플래그
                self._set_processing(False)

    def _save_wav_file(self, task):
        """WAV 파일 저장"""
        wav_path = os.path.join(
            config.TEMP_DIR_PATH,
            f"audio_{task['timestamp']:03d}_{task['session_id']}.wav",
        )
        os.makedirs(config.TEMP_DIR_PATH, exist_ok=True)

        logger.info(f"💾 원본 audio_bytes 크기: {len(task['audio_bytes'])} bytes")

        merged_wav = merge(task["audio_bytes"])
        logger.info(f"💾 합쳐진 WAV 크기: {len(merged_wav)} bytes")

        with open(wav_path, "wb") as f:
            f.write(merged_wav)

        file_size = os.path.getsize(wav_path)
        logger.info(f"💾 저장된 WAV 파일 크기: {file_size} bytes")

        return wav_path

    def _cleanup_wav_file(self, wav_path):
        """WAV 파일 정리"""
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logger.debug(f"WAV 파일 삭제: {wav_path}")
        except Exception as e:
            logger.warning(f"WAV 파일 삭제 실패 {wav_path}: {e}")

    async def add_chunk(self, audio_b64: str, frame_b64: str):
        """청크 추가 (완전 논블로킹)"""
        if self._shutdown:
            return None

        try:
            audio_bytes = base64.b64decode(audio_b64)
            self.buffer += audio_bytes
            logger.info(f"Session {self.session_id}: 오디오 데이터를 버퍼에 저장 중")

            # 프레임 업데이트
            if frame_b64:
                self.frame_latest = base64.b64decode(frame_b64)

            self.num_chunks += 1
            logger.info(
                f"Session {self.session_id}: 현재 num_chunks = {self.num_chunks}"
            )

            # 10초가 지났으면 데이터 queue에 정보 저장 후 초기화
            if self.should_process():
                self._enqueue_for_processing()

            # 만약 모델이 쉬고 있다면 process 진행
            self._try_start_model_processing()

            return None

        except Exception as e:
            logger.error(f"청크 추가 오류 (Session {self.session_id}): {e}")
            return None

    def _enqueue_for_processing(self):
        """처리할 데이터를 큐에 추가"""
        try:
            processing_data = {
                "audio_bytes": self.buffer,
                "frame": self.frame_latest,
                "timestamp": int(self.num_chunks),
                "session_id": self.session_id,
            }

            # 논블로킹으로 큐에 추가 (큐가 꽉 차면 가장 오래된 것 제거)
            try:
                self.data_queue.put_nowait(processing_data)
                logger.info(
                    f"Session {self.session_id}: 현재 data_queue 크기 = {self.data_queue.qsize()}"
                )
            except queue.Full:
                # 오래된 데이터 제거하고 새 데이터 추가
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(processing_data)
                    logger.warning(
                        f"Session {self.session_id}: 큐가 가득참, 오래된 데이터 제거"
                    )
                except queue.Empty:
                    pass

            # 버퍼 초기화
            self.buffer = b""
            self.frame_latest = None

        except Exception as e:
            logger.error(f"데이터 큐 추가 오류: {e}")

    def should_process(self) -> bool:
        return self.num_chunks % config.TIMESTEP == 0

    def _try_start_model_processing(self):
        """모델 처리 시작 시도 (스레드 안전)"""
        if self.is_processing() or self.data_queue.empty():
            return

        with self._processing_lock:
            if self._is_processing:  # 다시 한번 체크
                return

            try:
                processing_data = self.data_queue.get_nowait()
                self._is_processing = True

                # 모델 워커 스레드에게 작업 전달
                self.model_queue.put(processing_data)
                logger.info(
                    f"Session {self.session_id}: 모델 처리 작업을 워커 스레드에 전달"
                )

            except queue.Empty:
                return
            except Exception as e:
                logger.error(f"모델 처리 시작 오류: {e}")
                self._is_processing = False

    def cleanup(self):
        """리소스 정리"""
        logger.info(f"Session {self.session_id}: 정리 시작")

        # 종료 플래그 설정
        self._shutdown = True

        # 남은 작업들 완료 대기 (최대 10초)
        remaining_tasks = self.data_queue.qsize()
        if remaining_tasks > 0:
            logger.info(
                f"Session {self.session_id}: {remaining_tasks}개 작업 완료 대기 중..."
            )

        # 모델 워커 스레드 종료
        self.model_queue.put(None)
        if self.model_thread.is_alive():
            self.model_thread.join(timeout=10)
            if self.model_thread.is_alive():
                logger.warning(f"Session {self.session_id}: 워커 스레드 강제 종료")

        logger.info(f"Session {self.session_id}: 정리 완료")


# 전역 세션 관리
session_buffers = {}
executor = ThreadPoolExecutor(max_workers=4)  # 동시 처리 가능한 세션 수


# --- 웹소켓 엔드포인트 ---
@app.websocket("/ws/lecture-analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None

    if not analyzer:
        await websocket.send_json({"type": "error", "message": "서버 초기화 실패."})
        await websocket.close(code=1011)
        return

    try:
        while True:
            logger.info(f"세션 {session_id}: 메시지 수신 대기 중...")
            data = await websocket.receive_text()
            logger.info(f"세션 {session_id}: 메시지 수신됨. 데이터 길이: {len(data)}")
            message = json.loads(data)
            msg_type = message.get("type")
            logger.info(f"세션 {session_id}: {msg_type} 메시지 수신")

            # [수정] 종료 요청을 다른 어떤 메시지보다 먼저 확인하여 레이스 컨디션을 방지합니다.
            if msg_type == "end_session":
                logger.info(f"세션 종료 요청 받음: {session_id}")
                if not session_id:
                    await websocket.send_json(
                        {"type": "error", "message": "세션이 시작되지 않았습니다."}
                    )
                    break

                session = session_manager.get_session(session_id)
                if session:
                    logger.info(f"세션 '{session_id}'의 리포트 생성 시작...")
                    await websocket.send_json(
                        {
                            "type": "report_generating",
                            "message": "최종 리포트를 생성 중입니다...",
                        }
                    )

                    try:
                        final_report = await asyncio.to_thread(
                            analyzer.generate_final_report, session, session_id
                        )

                        logger.info(f"리포트 생성 완료, 전송 시작: {session_id}")
                        await websocket.send_json(
                            {"type": "final_report", "data": final_report}
                        )

                    except Exception as e:
                        logger.error(f"리포트 생성 중 에러 발생: {e}", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "message": f"리포트 생성 실패: {str(e)}"}
                        )
                else:
                    logger.warning(f"종료 요청된 세션을 찾을 수 없음: {session_id}")
                    await websocket.send_json(
                        {"type": "error", "message": "세션을 찾을 수 없습니다."}
                    )

                # 모든 작업 완료 후 세션 정리
                session_manager.remove_session(session_id)

                # 루프를 안전하게 종료
                break

            elif msg_type == "start_session":
                user_name = message.get("user_name", "학생")
                topic = message.get("topic", "학습 주제")
                session_id = session_manager.create_session(user_name, topic)
                await websocket.send_json(
                    {"type": "session_started", "session_id": session_id}
                )

            elif msg_type == "data_chunk":
                if not session_id:
                    await websocket.send_json(
                        {"type": "error", "message": "세션이 시작되지 않았습니다."}
                    )
                    continue

                session = session_manager.get_session(session_id)
                if session:
                    if session_id not in session_buffers:
                        session_buffers[session_id] = SessionAudioBuffer(
                            session_id, analyzer
                        )

                    buffer = session_buffers[session_id]

                    await buffer.add_chunk(
                        message.get("audio"),
                        message.get("frame"),
                    )

                    # --- Add counter for less frequent feedback ---
                    if "feedback_counter" not in session:
                        session["feedback_counter"] = 0
                    session["feedback_counter"] += 1

                    if (
                        session["feedback_counter"] % 5 == 0
                    ):  # Send feedback every 10 data_chunks
                        doc = collection.find_one(
                            {"session_id": session_id}, {"_id": 0, "results": 1}
                        )
                        results = doc.get("results", []) if doc else []
                        if results:
                            await websocket.send_json(
                                {
                                    "type": "realtime_feedback",
                                    "concentration": results[-1]["result"]["str"],
                                    "noise": results[-1]["noise"]["str"],
                                }
                            )
                    # --- End of counter logic ---

                    await asyncio.sleep(0.01)  # Increased sleep duration

    except WebSocketDisconnect:
        logger.info("웹소켓 연결이 끊어졌습니다.")
    except Exception as e:
        logger.error(f"웹소켓 오류 발생: {e}", exc_info=True)
    finally:
        if session_id and session_manager.get_session(session_id):
            logger.info(f"비정상 종료로 인한 세션 정리: {session_id}")
            session_manager.remove_session(session_id)


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if analyzer else "unhealthy",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("서버를 시작합니다. 주소: http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
