import asyncio
import json
from datetime import datetime
import io
import base64
import uuid
import logging
import os
from typing import Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- 프로젝트 모듈 임포트 ---
import config
import model_inference
from feedback_generator import GenerateFeedback
from data_process import analyze_concentration_changes

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
            "results": [],
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
            # End-to-End 멀티모달 모델 로드
            self.face_box_model, self.e2e_model = model_inference.load_model()
            self.e2e_model.to(self.device)
            logger.info("✅ 멀티모달 추론 모델 로드 완료")

        except Exception as e:
            logger.critical(f"❌ 모델 로딩 실패: {e}", exc_info=True)
            raise RuntimeError(
                "필수 모델 로딩에 실패하여 서버를 시작할 수 없습니다."
            ) from e

    def process_chunk(
        self, frame_data: bytes, audio_data: bytes, last_timestamp: float
    ) -> Dict | None:
        """실시간 데이터 청크를 받아 멀티모달 모델로 분석"""
        try:
            # 1. 데이터 전처리
            pil_image = Image.open(io.BytesIO(frame_data))
            dummy_audio_tensor = torch.randn(
                1, 1, 96, 64
            )  # 실제 오디오 데이터 처리 필요

            # 2. 모델 추론 실행
            (pred_num, pred_str), (yaw, pitch), (noise_num, noise_str) = (
                model_inference.run(
                    self.face_box_model, self.e2e_model, pil_image, dummy_audio_tensor
                )
            )

            # 3. 결과 구조화
            current_time = last_timestamp + (1.0 / config.VIDEO_FPS)
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

    def generate_final_report(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """세션 종료 시 종합 리포트 생성"""
        user_name = session_data["user_name"]
        topic = session_data["topic"]
        results = session_data["results"]

        if not results:
            logger.warning("분석할 데이터가 없어 리포트를 생성할 수 없습니다.")
            return {"error": "분석 데이터 부족"}

        logger.info(f"'{user_name}'님의 최종 리포트 생성 시작...")
        try:
            # 1. 피드백 생성 클래스 사용 (그래프 이미지가 HTML에 포함됨)
            feedback_generator = GenerateFeedback()
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
                            analyzer.generate_final_report, session
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
                    frame_b64 = message.get("frame")
                    audio_b64 = message.get("audio")

                    frame_data = base64.b64decode(frame_b64)
                    audio_data = base64.b64decode(audio_b64)

                    analysis_result = await asyncio.to_thread(
                        analyzer.process_chunk,
                        frame_data,
                        audio_data,
                        session["last_timestamp"],
                    )

                    if analysis_result:
                        session["results"].append(analysis_result)
                        logger.info(
                            f"세션 {session_id}: results 리스트 크기: {len(session['results'])}"
                        )
                        session["last_timestamp"] = analysis_result["timestamp"]["end"]

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
                                    "concentration": analysis_result["result"]["str"],
                                    "noise": analysis_result["noise"]["str"],
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
