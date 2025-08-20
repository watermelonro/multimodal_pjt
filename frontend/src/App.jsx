// App.jsx
import React, { useState, useEffect, useRef } from 'react';
import toWav from 'audiobuffer-to-wav';

// --- 백엔드 엔드포인트 ---
const API_URL = 'ws://localhost:8000/ws/lecture-analysis';
const BACKEND_URL = 'http://localhost:8000'; // (현재 미사용: 추후 REST 연동시 사용)

// --- Mock Data (백엔드에서 올 데이터 예시) ---
const mockSimplifiedTeacherData = {
  difficultTopics: [
    "전문가 시스템 (Expert Systems)의 개념 이해",
    "데이터베이스 정규화 과정의 복잡성",
  ],
  lowConcentrationPeriod: "강의 후반부 (30분 이후)",
  overallFeedback: "다수의 학생들이 추상적인 개념을 시각 자료 없이 이해하는 데 어려움을 겪고 있으며, 특히 긴 강의의 후반부에 집중력이 급격히 저하되는 경향을 보입니다. 짧은 복습 퀴즈나 쉬는 시간을 도입하는 것을 고려해볼 수 있습니다."
};

const mockStudentFeedback = {
    score: 82,
    summary: "전반적으로 훌륭한 집중력을 보여줬어요! 특히 '거래처리시스템' 파트에서는 시선과 자세가 매우 안정적이었습니다. 다만, 학습 후반부로 갈수록 약간의 피로감이 보였고, '전문가 시스템' 파트에서는 주변 소음에 잠시 집중이 흐트러지는 모습이 포착되었습니다.",
    positivePoints: ["안정적인 시선 처리", "높은 초기 몰입도"],
    improvementPoints: ["후반부 집중력 유지", "'전문가 시스템' 파트 복습"]
};

// --- 강의 주제 목록 ---
const lectureTopics = [
    "CHAPTER 1: 경영 정보 시스템",
    "CHAPTER 2: 의사 결정과 프로세스",
    "CHAPTER 3: e-비즈니스",
    "CHAPTER 4: 윤리와 정보 보호",
    "CHAPTER 5: 기반구조",
    "CHAPTER 6: 데이터",
    "CHAPTER 7: 네트워크",
    "CHAPTER 8: 전사적 애플리케이션",
    "CHAPTER 9: 시스템 개발과 프로젝트 관리",
];

// --- 재사용 가능한 UI 컴포넌트 ---
const DashboardCard = ({ title, children, className }) => (
  <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
    <h3 className="text-xl font-bold text-gray-800 mb-4">{title}</h3>
    <div className="space-y-3">{children}</div>
  </div>
);

const ChatWindow = ({ messages, onSendMessage }) => {
  const [input, setInput] = useState('');
  const handleSend = () => {
    if (input.trim()) {
      onSendMessage(input);
      setInput('');
    }
  };
  return (
    <div className="w-full h-[500px] bg-gray-50 rounded-lg p-4 flex flex-col">
      <div className="flex-grow space-y-4 overflow-y-auto pr-2">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl shadow-sm ${msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
              <p className="text-sm" style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex gap-2">
        <input
          type="text"
          placeholder="메시지를 입력하세요..."
          className="flex-grow p-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button onClick={handleSend} className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors">
          전송
        </button>
      </div>
    </div>
  );
};

// --- 로그인 화면 컴포넌트 ---
const LoginScreen = ({ onLogin }) => {
  const [studentId, setStudentId] = useState('');
  const [lectureTopic, setLectureTopic] = useState('');

  const handleLogin = () => {
    if (studentId.trim() && lectureTopic) {
      onLogin(studentId, lectureTopic);
    } else {
      alert('회원번호와 강의 주제를 모두 입력 및 선택해주세요.');
    }
  };

  return (
    <div className="text-center p-8 bg-white rounded-xl shadow-lg w-full max-w-md mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-8">학습자 정보 입력</h2>
      <div className="space-y-6">
        <input
          type="text"
          placeholder="회원번호를 입력하세요"
          value={studentId}
          onChange={(e) => setStudentId(e.target.value)}
          className="w-full p-4 border-2 border-gray-200 rounded-lg text-lg focus:outline-none focus:border-blue-500"
        />
        <select
          value={lectureTopic}
          onChange={(e) => setLectureTopic(e.target.value)}
          className={`w-full p-4 border-2 border-gray-200 rounded-lg text-lg focus:outline-none focus:border-blue-500 ${lectureTopic ? 'text-black' : 'text-gray-400'}`}
        >
          <option value="" disabled>강의 주제를 선택하세요</option>
          {lectureTopics.map(topic => (
            <option key={topic} value={topic} className="text-black">{topic}</option>
          ))}
        </select>
      </div>
      <button
        onClick={handleLogin}
        className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-4 px-8 rounded-lg text-lg mt-8 transition-transform transform hover:scale-105"
      >
        학습 시작하기
      </button>
    </div>
  );
};

// --- 페이지 컴포넌트 (백엔드 연결 포함) ---
const StudentView = () => {
  const [phase, setPhase] = useState('login');
  const [studentInfo, setStudentInfo] = useState(null);
  const [messages, setMessages] = useState([]);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // 전체 영상 로컬 녹화(다운로드용)
  const videoRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState(null);
  const [isCameraReady, setIsCameraReady] = useState(false);

  // --- WebSocket & 실시간 스트리밍 상태 ---
  const socketRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [realtimeFeedback, setRealtimeFeedback] = useState({ concentration: 'N/A', noise: 'N/A' });
  const [finalReport, setFinalReport] = useState(null);
  const [error, setError] = useState(null);

  // 오디오 청크 전송용 레코더(1초 타임슬라이스)
  const audioRecorderRef = useRef(null);

  // --- WebSocket 연결 (컴포넌트 마운트 시) ---
  useEffect(() => {
    const ws = new WebSocket(API_URL);

    ws.onopen = () => {
      // 연결 성공
      socketRef.current = ws;
      setError(null);
      // console.log('✅ WebSocket connected');
    };

    ws.onerror = (err) => {
      console.error('❌ WebSocket error:', err);
      setError('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.');
    };

    ws.onclose = () => {
      // console.log('🔒 WebSocket closed');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      // console.log('백엔드 메시지 수신:', message);

      switch (message.type) {
        case 'session_started':
          setSessionId(message.session_id);
          setIsStreaming(true);
          break;
        case 'realtime_feedback':
          setRealtimeFeedback(message);
          break;
        case 'report_generating':
          setPhase('loading_feedback');
          setIsStreaming(false);
          break;
        case 'final_report':
          setFinalReport(message.data);
          setPhase('choice');
          if (socketRef.current) socketRef.current.close();
          break;
        case 'error':
          setError(message.message);
          break;
        default:
          break;
      }
    };

    return () => {
      try { ws.close(); } catch (_) {}
    };
  }, []);

  // --- 카메라 준비/정리 ---
  useEffect(() => {
    if (phase === 'camera_setup' || phase === 'recording') {
      setIsCameraReady(false);
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          mediaStreamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
          setIsCameraReady(true);
        })
        .catch(err => {
          console.error('카메라 접근 오류:', err);
          setIsCameraReady(false);
          setError('카메라/마이크 권한이 필요합니다. 권한을 허용하고 페이지를 새로고침해주세요.');
        });
    } else {
      // phase 벗어나면 스트림 정리
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop());
        mediaStreamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  }, [phase]);

  // --- 오디오 청크 처리 후 WebSocket 전송 ---
  const processAudioChunk = async (audioBlob) => {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const wavBuffer = toWav(audioBuffer);
    const wavBytes = new Uint8Array(wavBuffer);
    const audioBase64 = btoa(String.fromCharCode(...wavBytes));

    // 현재 프레임 캡처 (jpeg base64)
    let frameBase64 = '';
    if (videoRef.current && videoRef.current.readyState >= 2) {
      const canvas = document.createElement('canvas');
      canvas.width = 1080;
      canvas.height = 720;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      frameBase64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: 'data_chunk',
        frame: frameBase64,
        audio: audioBase64
      }));
    }
  };

  // --- 스트리밍 시작 ---
  const startStreaming = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setError('서버와의 연결이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
      return;
    }

    // 세션 시작 알림
    socket.send(JSON.stringify({
      type: 'start_session',
      user_name: studentInfo?.id || '',   // 회원번호를 user_name 으로 전송
      topic: studentInfo?.topic || ''
    }));

    // 오디오 전용 레코더 시작 (1초 타임슬라이스)
    if (mediaStreamRef.current) {
      const audioStream = new MediaStream(mediaStreamRef.current.getAudioTracks());
      audioRecorderRef.current = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });

      audioRecorderRef.current.ondataavailable = async (e) => {
        if (e.data && e.data.size > 0) {
          try {
            await processAudioChunk(e.data);
          } catch (err) {
            console.error('오디오 청크 처리 오류:', err);
          }
        }
      };

      audioRecorderRef.current.start(1000); // 1초 간격 청크
      setIsStreaming(true);
    }
  };

  // --- 스트리밍 종료 ---
  const stopStreaming = () => {
    // 오디오 레코더 중지
    if (audioRecorderRef.current && audioRecorderRef.current.state !== 'inactive') {
      try { audioRecorderRef.current.stop(); } catch (_) {}
      audioRecorderRef.current = null;
    }

    // 세션 종료 메시지
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'end_session' }));
    }

    setIsStreaming(false);
    setPhase('loading_feedback');
  };

  // --- 로컬(전체) 비디오 녹화 컨트롤 ---
  const handleStartRecording = () => {
    if (!mediaStreamRef.current) return;
    // 전체 비디오 저장용 레코더
    const chunks = [];
    videoRecorderRef.current = new MediaRecorder(mediaStreamRef.current, { mimeType: 'video/webm' });
    videoRecorderRef.current.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    videoRecorderRef.current.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      setRecordedVideoUrl(url);
    };
    videoRecorderRef.current.start();

    // 실시간 분석 스트리밍 시작
    startStreaming();

    setIsRecording(true);
    setPhase('recording');
  };

  const handleStopRecording = () => {
    // 로컬 비디오 레코더 중지
    if (videoRecorderRef.current && videoRecorderRef.current.state !== 'inactive') {
      try { videoRecorderRef.current.stop(); } catch (_) {}
      videoRecorderRef.current = null;
    }
    setIsRecording(false);

    // 백엔드 스트리밍 종료
    stopStreaming();
    // phase 변경은 stopStreaming 내에서 loading_feedback 로 전환
  };

  const handleLogin = (id, topic) => {
    setStudentInfo({ id, topic });
    setPhase('camera_setup');
  };

  const setupChat = (mode) => {
    if (mode === 'test') {
      setMessages([{ sender: 'llm', text: "오늘 학습 내용 중 집중도가 낮았던 '전문가 시스템' 파트에 대해 간단한 문제를 내볼게요. 준비됐나요?" }]);
      setPhase('test');
    } else if (mode === 'feedback_chat') {
      setMessages([{ sender: 'llm', text: "오늘 학습 내용에 대해 더 궁금한 점이 있나요? '전문가 시스템' 파트가 어려웠다고 분석되었는데, 어떤 점이 가장 이해하기 힘들었나요?" }]);
      setPhase('feedback_chat');
    }
  };

  const handleSendMessage = (userMessage) => {
    const newMessages = [...messages, { sender: 'user', text: userMessage }];
    const botResponse = { sender: 'llm', text: "흥미로운 질문이네요! 그 부분에 대해 더 자세히 설명해 드릴게요..." };
    setMessages([...newMessages, botResponse]);
  };

  // --- 에러 화면 우선 표시 ---
  if (error) {
    return (
      <div className="text-center p-8 bg-red-100 text-red-700 rounded-xl shadow-lg w-full">
        <b>오류:</b> {error}
      </div>
    );
  }

  // --- phase 상태에 따라 화면 ---
  switch (phase) {
    case 'login':
      return <LoginScreen onLogin={handleLogin} />;

    case 'camera_setup':
    case 'recording':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg w-full">
          <div className="text-left mb-4 bg-gray-100 p-3 rounded-lg">
            <p className="text-sm text-gray-600"><strong>회원번호:</strong> {studentInfo?.id}</p>
            <p className="text-sm text-gray-600"><strong>강의주제:</strong> {studentInfo?.topic}</p>
          </div>

          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            {isRecording ? '학습 중...' : '학습 준비'}
          </h2>
          <p className="text-gray-600 mb-6">
            {isRecording ? '학습이 끝나면 "학습 종료" 버튼을 눌러주세요.' : '카메라가 켜지면 "학습 시작" 버튼을 눌러 녹화를 시작하세요.'}
          </p>

          <div className="relative w-full">
            <video
              ref={videoRef}
              autoPlay
              muted
              className="w-full bg-black aspect-video rounded-lg mb-6 transform -scale-x-100"
            />
            {isStreaming && (
              <div className="absolute bottom-4 left-4 bg-gray-900 bg-opacity-75 text-white text-base p-4 rounded-xl shadow-lg">
                <p className="font-bold">실시간 집중도: <span className="font-normal">{realtimeFeedback.concentration}</span></p>
                <p className="font-bold">주변 소음: <span className="font-normal">{realtimeFeedback.noise}</span></p>
              </div>
            )}
          </div>

          {!isRecording ? (
            <button
              onClick={handleStartRecording}
              disabled={!isCameraReady}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-lg text-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isCameraReady ? '학습 시작 및 녹화(스트리밍 포함)' : '카메라 준비 중...'}
            </button>
          ) : (
            <button
              onClick={handleStopRecording}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-8 rounded-lg text-lg"
            >
              학습 종료
            </button>
          )}
        </div>
      );

    case 'loading_feedback':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg animate-pulse">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">리포트 생성 중...</h2>
          <p className="text-gray-600 mb-8">학습 데이터를 분석하고 있습니다. 잠시만 기다려주세요.</p>
        </div>
      );

    case 'choice':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg animate-fade-in">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">학습 세션 완료!</h2>
          <p className="text-gray-600 mb-8">수고하셨습니다! 다음 단계를 선택해주세요.</p>
          {recordedVideoUrl && (
            <div className="mb-6">
              <a
                href={recordedVideoUrl}
                download={`${studentInfo.id}_${studentInfo.topic}.webm`}
                className="text-blue-600 hover:underline"
              >
                녹화된 학습 영상 다운로드
              </a>
            </div>
          )}
          <div className="flex justify-center gap-4">
            <button
              onClick={() => setupChat('test')}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-4 px-8 rounded-lg text-lg"
            >
              📝 테스트 보기
            </button>
            <button
              onClick={() => setPhase('feedback_summary')}
              className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-4 px-8 rounded-lg text-lg"
            >
              📊 피드백 보기
            </button>
          </div>
        </div>
      );

    case 'feedback_summary': {
      const hasFinal = !!finalReport?.llm_report;
      return (
        <div className="p-8 bg-white rounded-xl shadow-lg w-full animate-fade-in">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-bold text-gray-800">학습 피드백 요약</h2>
            <button
              onClick={() => setPhase('choice')}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg"
            >
              ← 이전으로
            </button>
          </div>

          {/* 실시간/최종 리포트(백엔드) */}
          {hasFinal && (
            <DashboardCard title="🤖 AI 종합 피드백(백엔드)">
              <div
                className="prose prose-sm max-w-none"
                dangerouslySetInnerHTML={{ __html: finalReport.llm_report }}
              />
            </DashboardCard>
          )}

          {/* 백업(모의 데이터) */}
          {!hasFinal && (
            <>
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                <div className="md:col-span-1 flex flex-col items-center justify-center bg-blue-50 p-6 rounded-lg">
                  <p className="text-lg text-blue-800 font-semibold">종합 집중도 점수</p>
                  <p className="text-7xl font-bold text-blue-600">{mockStudentFeedback.score}</p>
                </div>
                <div className="md:col-span-2 bg-gray-50 p-6 rounded-lg">
                  <h4 className="font-bold text-gray-700 mb-2">종합 분석</h4>
                  <p className="text-gray-600 text-sm leading-relaxed">{mockStudentFeedback.summary}</p>
                </div>
              </div>
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                <DashboardCard title="👍 잘했어요!" className="bg-green-50">
                  {mockStudentFeedback.positivePoints.map(point => <p key={point} className="text-green-800">✔ {point}</p>)}
                </DashboardCard>
                <DashboardCard title="💪 개선해봐요!" className="bg-orange-50">
                  {mockStudentFeedback.improvementPoints.map(point => <p key={point} className="text-orange-800">💡 {point}</p>)}
                </DashboardCard>
              </div>
            </>
          )}

          <div className="text-center">
            <button
              onClick={() => setupChat('feedback_chat')}
              className="bg-gray-700 hover:bg-gray-800 text-white font-bold py-3 px-8 rounded-lg text-lg"
            >
              💬 AI와 대화하며 더 알아보기
            </button>
          </div>
        </div>
      );
    }

    case 'test':
    case 'feedback_chat':
      return (
        <div className="w-full">
          <div className="text-right mb-2">
            <button
              onClick={() => setPhase('choice')}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg"
            >
              ← 이전으로
            </button>
          </div>
          <ChatWindow messages={messages} onSendMessage={handleSendMessage} />
        </div>
      );

    default:
      return null;
  }
};

// --- 교사 뷰 컴포넌트 (단순화된 버전) ---
const TeacherView = () => {
  const { difficultTopics, lowConcentrationPeriod, overallFeedback } = mockSimplifiedTeacherData;
  return (
    <div className="w-full max-w-4xl space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 text-center mb-6">교사용 종합 리포트</h2>
      <DashboardCard title="학생들이 어려워한 내용">
        <ul className="list-disc list-inside text-gray-700">
          {difficultTopics.map(topic => <li key={topic}>{topic}</li>)}
        </ul>
      </DashboardCard>
      <DashboardCard title="주요 집중도 저하 시점">
        <p className="text-gray-700 font-medium">{lowConcentrationPeriod}</p>
      </DashboardCard>
      <DashboardCard title="종합 피드백 및 교육 제안 (LLM)">
        <p className="text-gray-700 leading-relaxed">{overallFeedback}</p>
      </DashboardCard>
    </div>
  );
};

// --- 메인 앱 컴포넌트 ---
function App() {
  const [view, setView] = useState('student'); // 'student' or 'teacher'
  return (
    <div className="bg-gray-100 min-h-screen flex flex-col items-center justify-center font-sans p-4">
      <div className="mb-8 p-1 bg-gray-200 rounded-lg flex gap-1">
        <button
          onClick={() => setView('student')}
          className={`px-6 py-2 rounded-md font-semibold transition-colors ${view === 'student' ? 'bg-white shadow' : 'bg-transparent text-gray-600'}`}
        >
          학생 뷰
        </button>
        <button
          onClick={() => setView('teacher')}
          className={`px-6 py-2 rounded-md font-semibold transition-colors ${view === 'teacher' ? 'bg-white shadow' : 'bg-transparent text-gray-600'}`}
        >
          교사 뷰
        </button>
      </div>
      <div className="w-full max-w-4xl flex items-center justify-center">
        {view === 'student' ? <StudentView /> : <TeacherView />}
      </div>
    </div>
  );
}

export default App;
