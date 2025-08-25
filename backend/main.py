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
from pydantic import BaseModel
import openai 

import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        logger.info(f"분석기 초기화 중... (Device: {self.device})")
        try:
            self.pad = FastAudioPreprocessor(audio_conf)
            logger.info("✅ 음성 전처리 모듈 로드 완료")
            self.face_box_model, self.e2e_model = model_inference.load_model()
            self.e2e_model.to(self.device)
            model_inference.warmup_model(self.face_box_model, self.e2e_model)
            logger.info("✅ 멀티모달 추론 모델 로드 완료")
        except Exception as e:
            logger.critical(f"❌ 모델 로딩 실패: {e}", exc_info=True)
            raise RuntimeError("필수 모델 로딩에 실패하여 서버를 시작할 수 없습니다.") from e

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
            pil_image = Image.open(io.BytesIO(frame_data))
            pil_image.save("output.jpg")
            audio_tensor = preprocess_audio_data(self.pad, audio_path)
            ((pred_num, pred_str), (yaw, pitch), (noise_num, noise_str)), audio = (
                model_inference.run(
                    self.face_box_model, self.e2e_model, pil_image, audio_tensor
                )
            )
            transcribed_text = self._transcribe_audio(audio_path)
            start_time = last_timestamp - config.TIMESTEP
            result = {
                "timestamp": {"start": start_time, "end": last_timestamp},
                "result": {"num": pred_num, "str": pred_str},
                "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                "noise": {"num": noise_num, "str": noise_str},
                "text": transcribed_text,
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
            safe_user_name = "".join(c for c in user_name if c.isalnum())
            report_filename = f"feedback_{safe_user_name}_{uuid.uuid4().hex[:6]}.html"
            report_path_abs = os.path.join(config.STATIC_DIR, report_filename)
            with open(report_path_abs, "w", encoding="utf-8") as f:
                f.write(full_html_report)
            logger.info(f"✅ HTML 리포트 파일 저장 완료: {report_path_abs}")
            collection.update_one(
                {"session_id": session_id},
                {"$set": {"final_report_html": full_html_report}}
            )
            logger.info(f"✅ 최종 리포트를 데이터베이스에 저장했습니다.")
            sorted_results = sorted(results, key=lambda x: x["timestamp"]["start"])
            insights = analyze_concentration_changes(sorted_results)
            logger.info("✅ 데이터 심층 분석 완료")
            return {
                "user_name": user_name,
                "topic": topic,
                "llm_report": full_html_report,
                "detailed_analysis": insights,
            }
        except Exception as e:
            logger.error(f"최종 리포트 생성 중 오류 발생: {e}", exc_info=True)
            return {"error": "리포트 생성 중 서버 오류 발생"}

# --- 전역 분석기 인스턴스 생성 ---
try:
    analyzer = LectureAnalyzer()
    llm_pipeline = LLMPipeline()
    logger.info("✅ LLM Pipeline 로드 완료")
except RuntimeError as e:
    logger.critical(f"분석기 인스턴스 생성 실패. 서버를 종료합니다. 오류: {e}")
    analyzer = None

# class SessionAudioBuffer:
#     def __init__(self, session_id: str, analyzer):
#         self.session_id = session_id
#         self.analyzer = analyzer
#         self.num_chunks = 0
#         self._shutdown = False
#         self.buffer = b''
#         self.frame_latest = None
#         self.model_queue = Queue()
#         self.model_thread = threading.Thread(target=self._model_worker)
#         self.model_thread.start()

#     def _model_worker(self):
#         """모델 추론 전용 워커 스레드. 큐에 들어온 작업을 순차적으로 처리합니다."""
#         while True:
#             task = self.model_queue.get()
#             if task is None:  # 종료 신호
#                 self.model_queue.task_done()
#                 break
            
#             wav_path = ""
#             try:
#                 wav_path = self._save_wav_file(task)
#                 result = self.analyzer.process_chunk(
#                     task['frame'], wav_path, task['timestamp']
#                 )
#                 if result:
#                     save_result(self.session_id, result)
#             except Exception as e:
#                 logger.error(f"모델 워커 처리 중 오류 발생 (Session {self.session_id}): {e}", exc_info=True)
#             finally:
#                 self._cleanup_wav_file(wav_path)
#                 self.model_queue.task_done()
class SessionAudioBuffer:
    def __init__(self, session_id: str, analyzer):
        self.session_id = session_id
        self.analyzer = analyzer
        self.num_chunks = 0
        self._shutdown = False
        self.buffer = b''
        self.frame_latest = None
        self.model_queue = Queue()
        
        # 배치 처리만 사용
        self.batch_size = 3
        self.batch_buffer = []
        
        self.model_thread = threading.Thread(target=self._model_worker)
        self.model_thread.start()

    def _model_worker(self):
        """배치 처리만 하는 워커 (기존 개별 처리 제거)"""
        while True:
            try:
                task = self.model_queue.get(timeout=1)
                if task is None:  # 종료 신호
                    if self.batch_buffer:
                        asyncio.run(self._process_batch())
                    break
                
                # 배치에 추가
                self.batch_buffer.append(task)
                
                # 배치가 찼거나 shutdown이면 처리
                if len(self.batch_buffer) >= self.batch_size or self._shutdown:
                    asyncio.run(self._process_batch())
                
                self.model_queue.task_done()
                
            except queue.Empty:
                if self._shutdown and self.batch_buffer:
                    asyncio.run(self._process_batch())
                continue

    async def _process_batch(self):
        """배치 병렬 처리"""
        if not self.batch_buffer:
            return
            
        current_batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        
        logger.info(f"Session {self.session_id}: 배치 처리 시작 ({len(current_batch)}개)")
        
        # 배치 내 모든 작업을 병렬로 처리
        batch_tasks = []
        for task in current_batch:
            task_coroutine = self._process_single_task_async(task)
            batch_tasks.append(task_coroutine)
        
        # 순서 보장 병렬 처리
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 결과 저장
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"배치 오류: {result}")
            elif result:
                save_result(self.session_id, result)
                
        logger.info(f"Session {self.session_id}: 배치 완료")

    async def _process_single_task_async(self, task):
        """개별 작업 비동기 처리"""
        wav_path = ""
        try:
            wav_path = self._save_wav_file(task)
            
            # 비동기 청크 처리
            result = await self._process_chunk_async(
                task['frame'], wav_path, task['timestamp']
            )
            return result
            
        except Exception as e:
            logger.error(f"개별 작업 오류: {e}")
            return None
        finally:
            self._cleanup_wav_file(wav_path)

    async def _process_chunk_async(self, frame_data, audio_path, timestamp):
        """비동기 청크 처리 (STT만 비동기, 나머지는 동기)"""
        try:
            # 동기 처리 부분
            pil_image = Image.open(io.BytesIO(frame_data))
            pil_image.save("output.jpg")
            audio_tensor = preprocess_audio_data(self.analyzer.pad, audio_path)
            ((pred_num, pred_str), (yaw, pitch), (noise_num, noise_str)), audio = (
                model_inference.run(
                    self.analyzer.face_box_model, 
                    self.analyzer.e2e_model, 
                    pil_image, 
                    audio_tensor
                )
            )
            
            # 비동기 STT 처리 - 핵심!
            transcribed_text = await self._async_whisper_call(audio_path)
            
            start_time = timestamp - config.TIMESTEP
            result = {
                "timestamp": {"start": start_time, "end": timestamp},
                "result": {"num": pred_num, "str": pred_str},
                "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                "noise": {"num": noise_num, "str": noise_str},
                "text": transcribed_text,
            }
            return result
            
        except Exception as e:
            logger.error(f"청크 처리 오류: {e}")
            return None

    async def _async_whisper_call(self, audio_path: str) -> str:
        """Whisper API 비동기 호출"""
        def whisper_sync_call():
            try:
                with open(audio_path, "rb") as audio_file:
                    transcription = self.analyzer.openai_client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        timeout=10
                    )
                return transcription.text
            except Exception as e:
                logger.warning(f"Whisper 오류: {e}")
                return "[음성인식 실패]"
        
        # 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            text = await loop.run_in_executor(executor, whisper_sync_call)
        return text
    def _save_wav_file(self, task):
        wav_path = os.path.join(
            config.TEMP_DIR_PATH, 
            f"audio_{task['timestamp']:03d}_{self.session_id}.wav"
        )
        os.makedirs(config.TEMP_DIR_PATH, exist_ok=True)
        merged_wav = merge(task['audio_bytes'])
        with open(wav_path, "wb") as f:
            f.write(merged_wav)
        return wav_path

    def _cleanup_wav_file(self, wav_path):
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"WAV 파일 삭제 실패 {wav_path}: {e}")

    async def add_chunk(self, audio_b64: str, frame_b64: str):
        if self._shutdown:
            return
        try:
            self.buffer += base64.b64decode(audio_b64)
            if frame_b64:
                self.frame_latest = base64.b64decode(frame_b64)
            self.num_chunks += 1
            if self.should_process():
                self._enqueue_for_processing()
        except Exception as e:
            logger.error(f"청크 추가 오류 (Session {self.session_id}): {e}")

    def _enqueue_for_processing(self):
        try:
            processing_data = {
                "audio_bytes": self.buffer,
                "frame": self.frame_latest,
                "timestamp": int(self.num_chunks),
                "session_id": self.session_id
            }
            self.model_queue.put(processing_data)
            self.buffer = b''
            self.frame_latest = None
        except Exception as e:
            logger.error(f"데이터 큐 추가 오류: {e}")

    def should_process(self) -> bool:
        return self.num_chunks % config.TIMESTEP == 0

    def shutdown(self):
        """모든 큐 작업을 완료하고 스레드를 정상적으로 종료합니다."""
        logger.info(f"Session {self.session_id}: Shutdown initiated.")
        self._shutdown = True
        if self.buffer:
            self._enqueue_for_processing()
        
        logger.info(f"Session {self.session_id}: Waiting for all tasks to be processed...")
        self.model_queue.join()  # 큐의 모든 항목이 처리될 때까지 블로킹
        
        logger.info(f"Session {self.session_id}: All tasks processed. Stopping worker thread.")
        self.model_queue.put(None) # 종료 신호 전송
        self.model_thread.join()
        logger.info(f"Session {self.session_id}: Shutdown complete.")

# 전역 세션 관리
session_buffers = {}

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

            if msg_type == "start_session":
                user_name = message.get("user_name", "학생")
                topic = message.get("topic", "학습 주제")
                session_id = session_manager.create_session(user_name, topic)
                session_buffers[session_id] = SessionAudioBuffer(session_id, analyzer)
                await websocket.send_json({"type": "session_started", "session_id": session_id})

            elif msg_type == "data_chunk":
                if session_id and session_id in session_buffers:
                    await session_buffers[session_id].add_chunk(
                        message.get("audio"), message.get("frame")
                    )
                    # 실시간 피드백을 배치 크기의 배수로 조정
                    session = session_manager.get_session(session_id)
                    if session:
                        if "feedback_counter" not in session:
                            session["feedback_counter"] = 0
                        session["feedback_counter"] += 1

                        # 🔥 배치 크기(3) * 2 = 6청크마다 피드백 (배치 완료 후 조회)
                        if session["feedback_counter"] % 6 == 0:
                            # 잠시 기다렸다가 조회 (배치 처리 완료 대기)
                            await asyncio.sleep(1)  
                            
                            doc = collection.find_one({"session_id": session_id}, {"_id": 0, "results": 1})
                            results = doc.get("results", []) if doc else []
                            if results:
                                await websocket.send_json({
                                    "type": "realtime_feedback",
                                    "concentration": results[-1]['result']['str'],
                                    "noise": results[-1]['noise']['str'],
                                })
                    # # --- 실시간 피드백 전송 로직 ---
                    # session = session_manager.get_session(session_id)
                    # if session:
                    #     if "feedback_counter" not in session:
                    #         session["feedback_counter"] = 0
                    #     session["feedback_counter"] += 1

                    #     if session["feedback_counter"] % 5 == 0: # 5 청크마다 피드백 전송
                    #         doc = collection.find_one({"session_id": session_id}, {"_id": 0, "results": 1})
                    #         results = doc.get("results", []) if doc else []
                    #         if results:
                    #             await websocket.send_json(
                    #                 {
                    #                     "type": "realtime_feedback",
                    #                     "concentration": results[-1]['result']['str'],
                    #                     "noise": results[-1]['noise']['str'],
                    #                 }
                    #             )

            elif msg_type == "end_session":
                if not session_id:
                    await websocket.send_json({"type": "error", "message": "세션이 시작되지 않았습니다."})
                    break
                
                if session_id in session_buffers:
                    buffer = session_buffers[session_id]
                    await websocket.send_json({"type": "status_update", "message": "Final audio processing..."})
                    await asyncio.to_thread(buffer.shutdown)
                    del session_buffers[session_id]

                session = session_manager.get_session(session_id)
                if session:
                    logger.info(f"세션 '{session_id}'의 리포트 생성 시작...")
                    await websocket.send_json({"type": "report_generating", "message": "최종 리포트를 생성 중입니다..."})
                    try:
                        final_report = await asyncio.to_thread(
                            analyzer.generate_final_report, session, session_id
                        )
                        await websocket.send_json({"type": "final_report", "data": final_report})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"리포트 생성 실패: {str(e)}"})
                
                session_manager.remove_session(session_id)
                break
            
    except WebSocketDisconnect:
        logger.info(f"웹소켓 연결이 끊어졌습니다: {session_id}")
    finally:
        if session_id and session_id in session_buffers:
            session_buffers[session_id].shutdown()
            del session_buffers[session_id]
        if session_id and session_manager.get_session(session_id):
            session_manager.remove_session(session_id)

@app.post("/api/chat")
async def chat_with_teacher(request: ChatRequest):
    try:
        doc = collection.find_one({"session_id": request.session_id}, {"_id": 0, "results": 1, "final_report_html": 1})
        if not doc:
            return {"success": False, "response": "세션 데이터를 찾을 수 없습니다."}
        results = doc.get("results", [])
        final_report = doc.get("final_report_html", "")
        ai_response = llm_pipeline.generate_chat_response(
            user_message=request.message,
            user_name=request.user_name,
            topic=request.topic,
            analysis_results=results,
            final_report=final_report
        )
        return {"success": True, "response": ai_response}
    except Exception as e:
        logger.error(f"채팅 API 오류: {e}")
        return {"success": False, "response": "답변 생성 중 오류가 발생했습니다."}

@app.get("/health")
def health_check():
    return {"status": "healthy" if analyzer else "unhealthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)