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
from pydantic import BaseModel
import openai  # <-- OpenAI 라이브러리 임포트

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
from wav_process import FastAudioPreprocessor, preprocess_audio_data
from merge_wav import merge_wav_chunks_from_buffer as merge
from llm_prompt import LLMPipeline
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

audio_conf = {
    'num_mel_bins': 128, 
    'target_length': 1024, 
    'freqm': 48, 
    'timem': 192,  
    'dataset': 'aihub_audio_dataset', 
    'mean':-4.2677393, 
    'std':4.5689974
    }

def save_result(sessionid, result):
    collection.update_one(
        {"session_id": sessionid},
        {"$push": {"results": result}},
        upsert=True
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

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_name: str = "학생"
    topic: str = "경영정보시스템"
    
# --- 핵심 분석기 클래스 ---
class LectureAnalyzer:
    """모든 모델을 총괄하고 데이터 분석 파이프라인을 실행"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # OpenAI 클라이언트 초기화
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        logger.info(f"분석기 초기화 중... (Device: {self.device})")
        try:
            self.pad = FastAudioPreprocessor(audio_conf)
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

    def _transcribe_audio(self, audio_path: str) -> str:
        """오디오 파일을 Whisper API를 통해 텍스트로 변환"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            logger.info(f"Whisper 변환 완료: {transcription.text}")
            return transcription.text
        except Exception as e:
            logger.error(f"Whisper API 호출 중 오류 발생: {e}")
            return "음성 인식 중 오류가 발생했습니다."

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

            # 2-1. Whisper로 음성->텍스트 변환
            transcribed_text = self._transcribe_audio(audio_path)

            # 3. 결과 구조화
            start_time = last_timestamp - config.TIMESTEP
            result = {
                "timestamp": {"start": start_time, "end": last_timestamp},
                "result": {"num": pred_num, "str": pred_str},
                "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                "noise": {"num": noise_num, "str": noise_str},
                "text": transcribed_text, # Whisper로 변환된 실제 텍스트로 대체
            }
            return result

        except Exception as e:
            logger.error(f"실시간 처리 중 오류 발생: {e}", exc_info=True)
            return None

    def generate_final_report(self, session_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """세션 종료 시 종합 리포트 생성"""
        user_name = session_data["user_name"]
        topic = session_data["topic"]

        logger.info(f"'{user_name}'님의 최종 리포트 생성 시작...")
        try:
            # 1. 피드백 생성 클래스 사용 (그래프 이미지가 HTML에 포함됨)
            feedback_generator = GenerateFeedback()
            doc = collection.find_one({"session_id": session_id}, {"_id": 0, "results": 1})
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

            # 3. 생성된 리포트를 DB에 저장하여 채팅에 활용
            try:
                collection.update_one(
                    {"session_id": session_id},
                    {"$set": {"final_report_html": full_html_report}}
                )
                logger.info(f"✅ 최종 리포트를 데이터베이스에 저장했습니다.")
            except Exception as e:
                logger.error(f"최종 리포트 데이터베이스 저장 중 오류 발생: {e}")

            # 4. 심층 분석 데이터 (선택 사항이지만 유지)
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
    llm_pipeline = LLMPipeline()
    logger.info("✅ LLM Pipeline 로드 완료")
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
        self.buffer = b''
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
        while not self._shutdown or not self.model_queue.empty():
            try:
                task = self.model_queue.get(timeout=0.1)
                if task is None:  # 종료 신호
                    break
                
                start_time = time.time()
                logger.info(f"Session {task['session_id']}: 모델 추론 시작")
                
                # WAV 파일 저장
                wav_path = self._save_wav_file(task)
                
                try:
                    # 모델 추론
                    result = self.analyzer.process_chunk(
                        task['frame'], wav_path, task['timestamp']
                    )
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Session {task['session_id']}: 모델 추론 완료 ({processing_time:.2f}초)")

                    # MongoDB에 저장
                    if result:
                        save_result(self.session_id, result)
                    
                except Exception as e:
                    logger.error(f"모델 추론 오류 (Session {task['session_id']}): {e}")
                finally:
                    # 파일 정리
                    self._cleanup_wav_file(wav_path)
                
            except queue.Empty:
                if self._shutdown:
                    break
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
            f"audio_{task['timestamp']:03d}_{task['session_id']}.wav"
        )
        os.makedirs(config.TEMP_DIR_PATH, exist_ok=True)
        
        logger.info(f"💾 원본 audio_bytes 크기: {len(task['audio_bytes'])} bytes")
        
        merged_wav = merge(task['audio_bytes'])
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
            logger.warning(f"Session {self.session_id}: Shutdown in progress, chunk ignored.")
            return
            
        try:
            audio_bytes = base64.b64decode(audio_b64)
            self.buffer += audio_bytes
            logger.info(f"Session {self.session_id}: 오디오 데이터를 버퍼에 저장 중")

            # 프레임 업데이트
            if frame_b64:
                self.frame_latest = base64.b64decode(frame_b64)
            
            self.num_chunks += 1
            logger.info(f"Session {self.session_id}: 현재 num_chunks = {self.num_chunks}")
            
            # 10초가 지났으면 데이터 queue에 정보 저장 후 초기화
            if self.should_process():
                self._enqueue_for_processing()

            # 만약 모델이 쉬고 있다면 process 진행
            self._try_start_model_processing()
            
        except Exception as e:
            logger.error(f"청크 추가 오류 (Session {self.session_id}): {e}")
        
    def _enqueue_for_processing(self):
        """처리할 데이터를 큐에 추가"""
        try:
            # 마지막 청크일 경우, 버퍼에 남은 모든 데이터를 처리
            if self._shutdown and self.buffer:
                logger.info(f"Session {self.session_id}: Finalizing remaining buffer data.")
            
            processing_data = {
                "audio_bytes": self.buffer,
                "frame": self.frame_latest,
                "timestamp": int(self.num_chunks),
                "session_id": self.session_id
            }
            
            self.data_queue.put(processing_data, block=False)
            logger.info(f"Session {self.session_id}: 현재 data_queue 크기 = {self.data_queue.qsize()}")
            
            # 버퍼 초기화
            self.buffer = b""
            self.frame_latest = None
            
        except queue.Full:
            logger.warning(f"Session {self.session_id}: 큐가 가득참, 데이터 처리 지연 가능성 있음")
        except Exception as e:
            logger.error(f"데이터 큐 추가 오류: {e}")
    
    def should_process(self) -> bool:
        return self.num_chunks % config.TIMESTEP == 0
    
    def _try_start_model_processing(self):
        """모델 처리 시작 시도 (스레드 안전)"""
        if self.is_processing() or self.data_queue.empty():
            return
        
        with self._processing_lock:
            if self._is_processing:
                return
                
            try:
                processing_data = self.data_queue.get_nowait()
                self._is_processing = True
                self.model_queue.put(processing_data)
                logger.info(f"Session {self.session_id}: 모델 처리 작업을 워커 스레드에 전달")
                
            except queue.Empty:
                return
            except Exception as e:
                logger.error(f"모델 처리 시작 오류: {e}")
                self._is_processing = False
    
    def shutdown(self):
        """Gracefully process remaining data and shut down the worker."""
        logger.info(f"Session {self.session_id}: Graceful shutdown initiated.")
        self._shutdown = True

        # Enqueue any remaining data from the buffer
        if self.buffer:
            logger.info(f"Session {self.session_id}: Enqueuing final buffer chunk.")
            self._enqueue_for_processing()

        # Wait for the data queue to be processed by the model queue
        while not self.data_queue.empty():
            self._try_start_model_processing()
            time.sleep(0.2)

        # Wait for the model processing queue to be empty
        while not self.model_queue.empty() or self.is_processing():
            time.sleep(0.2)

        logger.info(f"Session {self.session_id}: All queued tasks processed.")

        # Now, stop the worker thread
        self.model_queue.put(None)
        if self.model_thread.is_alive():
            self.model_thread.join(timeout=20)
            if self.model_thread.is_alive():
                logger.warning(f"Session {self.session_id}: Worker thread did not terminate gracefully.")

        logger.info(f"Session {self.session_id}: Shutdown complete.")

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
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if session_id:
                logger.info(f"세션 {session_id}: {msg_type} 메시지 수신")
            else:
                logger.info(f"세션 없음: {msg_type} 메시지 수신")

            if msg_type == "start_session":
                user_name = message.get("user_name", "학생")
                topic = message.get("topic", "학습 주제")
                session_id = session_manager.create_session(user_name, topic)
                session_buffers[session_id] = SessionAudioBuffer(session_id, analyzer)
                await websocket.send_json(
                    {"type": "session_started", "session_id": session_id}
                )

            elif msg_type == "data_chunk":
                if session_id and session_id in session_buffers:
                    buffer = session_buffers[session_id]
                    await buffer.add_chunk(
                        message.get("audio"),
                        message.get("frame"),
                    )
                else:
                    logger.warning(f"Invalid or missing session_id for data_chunk.")

            elif msg_type == "end_session":
                logger.info(f"세션 종료 요청 받음: {session_id}")
                if not session_id:
                    await websocket.send_json({"type": "error", "message": "세션이 시작되지 않았습니다."})
                    break

                # 1. Gracefully shut down the session buffer
                if session_id in session_buffers:
                    buffer = session_buffers[session_id]
                    logger.info(f"세션 버퍼 종료 시작: {session_id}")
                    await asyncio.to_thread(buffer.shutdown)
                    logger.info(f"세션 버퍼 종료 완료: {session_id}")
                    del session_buffers[session_id]

                # 2. Generate the final report
                session = session_manager.get_session(session_id)
                if session:
                    logger.info(f"세션 '{session_id}'의 리포트 생성 시작...")
                    await websocket.send_json(
                        {"type": "report_generating", "message": "최종 리포트를 생성 중입니다..."}
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

                # 3. Clean up session from manager
                session_manager.remove_session(session_id)
                logger.info(f"웹소켓 루프를 종료합니다: {session_id}")
                break  # End the while loop
            
    except WebSocketDisconnect:
        logger.info(f"웹소켓 연결이 끊어졌습니다: {session_id}")
    except Exception as e:
        logger.error(f"웹소켓 오류 발생: {e}", exc_info=True)
    finally:
        # Final cleanup on disconnect
        if session_id:
            if session_id in session_buffers:
                logger.info(f"비정상 종료로 인한 세션 버퍼 정리: {session_id}")
                # Note: a full shutdown might be too slow here
                del session_buffers[session_id]
            if session_manager.get_session(session_id):
                logger.info(f"비정상 종료로 인한 세션 매니저 정리: {session_id}")
                session_manager.remove_session(session_id)

@app.post("/api/chat")
async def chat_with_teacher(request: ChatRequest):
    try:
        # MongoDB에서 세션 데이터 조회 (리포트 포함)
        doc = collection.find_one(
            {"session_id": request.session_id}, 
            {"_id": 0, "results": 1, "final_report_html": 1}
        )
        
        if not doc:
            return {"success": False, "response": "세션 데이터를 찾을 수 없습니다."}
        
        results = doc.get("results", [])
        final_report = doc.get("final_report_html", "") # 최종 리포트 추출
        
        # RAG Pipeline 사용
        ai_response = llm_pipeline.generate_chat_response(
            user_message=request.message,
            user_name=request.user_name,
            topic=request.topic,
            analysis_results=results,
            final_report=final_report # 최종 리포트 전달
        )
        
        return {"success": True, "response": ai_response}
        
    except Exception as e:
        logger.error(f"채팅 API 오류: {e}")
        return {"success": False, "response": "답변 생성 중 오류가 발생했습니다."}
    
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
