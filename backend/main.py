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
    allow_origins=["http://localhost:5173"],  # 개발 중에는 모든 오리진 허용
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
            "last_timestamp": 0.0,
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
            self.pad = load_preprocessor(os.path.join(config.MODELS_DIR, 'train_dataset_scaler_gpu.pkl'))
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

            audio_tensor = preprocess_audio_data(self.pad, audio_path)

            # 2. 모델 추론 실행
            (pred_num, pred_str), (yaw, pitch), (noise_num, noise_str) = (
                model_inference.run(
                    self.face_box_model, self.e2e_model, pil_image, audio_tensor
                )
            )

            # 3. 결과 구조화
            current_time = last_timestamp + config.TIMESTEP
            result = {
                "timestamp": {"start": last_timestamp, "end": current_time},
                "result": {"num": pred_num, "str": pred_str},
                "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                "noise": {"num": noise_num, "str": noise_str},
                "text": f"({self._format_time(current_time)} 지점의 강의 내용) ",  # Whisper 연동 시 실제 텍스트로 대체
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
            doc = collection.find_one({"sessionid": session_id}, {"_id": 0, "results": 1})
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
        self.session_id = session_id
        self.current_buffer = b''
        self.backup_buffer = b''  # 처리 중일 때 사용할 백업 버퍼
        self.buffer_select = 0
        self.num_chunks = 0
        self.frame_latest = None
        self.is_processing = False
        self.processing_queue = Queue()
        self.analyzer = analyzer
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pending_results = []  # 대기 중인 결과들
        
    def add_chunk(self, audio_b64: str, frame_b64: str):
        """청크 추가 (논블로킹)"""
        audio_bytes = base64.b64decode(audio_b64)

        if self.is_processing:
            # 처리 중이면 백업 버퍼에 저장
            self.backup_buffer += audio_bytes
            logger.info(f"Session {self.session_id}: 처리 중이므로 백업 버퍼에 저장")
        else:
            # 평상시에는 메인 버퍼에 저장
            self.current_buffer += audio_bytes

        # 프레임 업데이트
        if frame_b64:
            self.frame_latest = base64.b64decode(frame_b64)
        
        self.num_chunks += 1
        
        # 10초가 지났고 처리 중이 아니면 백그라운드에서 처리 시작
        if self.should_process() and not self.is_processing:
            future = self.start_background_processing()
            return future
        
        return None
    
    def should_process(self) -> bool:
        return self.num_chunks % config.TIMESTEP == 0
    
    def start_background_processing(self):
        """백그라운드에서 처리 시작"""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        # 현재 버퍼를 처리 대상으로 넘기고, 새 버퍼 시작
        processing_data = {
            'audio_bytes': self.current_buffer,
            'frame': self.frame_latest,
            'timestamp' : int(self.num_chunks//config.TIMESTEP - 1),
            'session_id': self.session_id
        }
        
        # 백업 버퍼를 메인으로 이동 (처리 중 쌓인 데이터)
        self.current_buffer = self.backup_buffer
        self.backup_buffer = b''
        self.frame_latest = None
        
        # 백그라운드 스레드에서 처리
        future = self.executor.submit(self._process_background, processing_data)

        future.add_done_callback(self._on_processing_complete)
        
        logger.info(f"Session {self.session_id}: 백그라운드 처리 시작, 새 버퍼로 계속 수집")
        return future
    
    def _process_background(self, data):
        """백그라운드에서 실제 처리"""
        try:
            wav_path = os.path.join(config.TEMP_DIR_PATH, f"audio_{data['timestamp']:03d}_{data['session_id']}.wav")
            
            os.makedirs(config.TEMP_DIR_PATH, exist_ok=True)

            logger.info(f"💾 원본 audio_bytes 크기: {len(data['audio_bytes'])} bytes")

            merged_wav = merge(data['audio_bytes'])
            logger.info(f"💾 합쳐진 WAV 크기: {len(merged_wav)} bytes")

            # WAV 파일 저장
            with open(wav_path, "wb") as f:
                f.write(merged_wav)

            # 파일 크기 확인
            file_size = os.path.getsize(wav_path)
            logger.info(f"💾 저장된 WAV 파일 크기: {file_size} bytes")
            
            # WAV 파일 길이 확인
            import wave
            try:
                with wave.open(wav_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    logger.info(f"🎵 최종 WAV 파일 길이: {duration:.2f}초")  # 이제 10초 나와야 함!
            except Exception as e:
                logger.error(f"WAV 파일 분석 실패: {e}")
            
            logger.info(f"Session {data['session_id']}: 처리 시작)")
            
            # 시간이 오래 걸리는 모델 처리
            result = self.analyzer.process_chunk(data['frame'], wav_path, data['timestamp'])  # 이게 5-10초 걸려도 OK
            
            logger.info(f"Session {data['session_id']}: 처리 완료 - {result}")
            
            # 파일 정리
            """
            os.remove(wav_path)
            """
            
            return {
                'session_id': data['session_id'],
                'result': result,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"백그라운드 처리 오류: {e}")
        finally:
            # 처리 완료 플래그
            self.is_processing = False

    def _on_processing_complete(self, future):
        """처리 완료 콜백"""
        try:
            result = future.result()
            if result['success']:
                logger.info(f"처리 성공: {result['result']}")
                # 여기서 결과를 저장하거나 전송
                self.handle_result(result)
            else:
                logger.error(f"처리 실패: {result['error']}")
        except Exception as e:
            logger.error(f"결과 처리 중 오류: {e}")

    def handle_result(self, result):
        """결과 처리 - 오버라이드 가능"""
        # 결과를 큐에 저장하거나 즉시 처리
        self.processing_queue.put(result)

    def get_latest_results(self):
        """완료된 결과들 가져오기"""
        results = []
        while not self.processing_queue.empty():
            try:
                result = self.processing_queue.get_nowait()
                results.append(result)
            except:
                break
        return results

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
                        session_buffers[session_id] = SessionAudioBuffer(session_id, analyzer)

                    buffer = session_buffers[session_id]

                    future = buffer.add_chunk(
                        message.get("audio"),
                        message.get("frame"),
                    )

                    """
                    future = {
                        'session_id': data['session_id'],
                        'result': result,
                        'success': True
                    }

                    result = {
                        "timestamp": {"start": last_timestamp, "end": current_time},
                        "result": {"num": pred_num, "str": pred_str},
                        "pose": {"yaw": float(yaw), "pitch": float(pitch)},
                        "noise": {"num": noise_num, "str": noise_str},
                        "text": f"({self._format_time(current_time)} 지점의 강의 내용) ",  # Whisper 연동 시 실제 텍스트로 대체
                    }
                    """
                    # 즉시 결과 필요하면 (블로킹)
                    if future:
                        result = future.result()  # 처리 완료까지 기다림

                        save_result(session_id, result)
                        logger.info(
                            f"세션 {session_id}: results 리스트 크기: {len(session['results'])}"
                        )
                        session["last_timestamp"] = result["timestamp"]["end"]

                        # --- Add counter for less frequent feedback ---
                        if "feedback_counter" not in session:
                            session["feedback_counter"] = 0
                        session["feedback_counter"] += 1

                        if (
                            session["feedback_counter"] % 3 == 0
                        ):  # Send feedback every 3 data_chunks
                            await websocket.send_json(
                                {
                                    "type": "realtime_feedback",
                                    "concentration": result["result"]["str"],
                                    "noise": result["noise"]["str"],
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
