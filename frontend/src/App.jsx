import React, { useState, useEffect, useRef } from 'react';
import toWav from 'audiobuffer-to-wav';

const API_URL = 'ws://localhost:8000/ws/lecture-analysis';
const BACKEND_URL = 'http://localhost:8000';

// --- 강의 주제 목록 (새로 추가) ---
const lectureTopics = [
    "CHAPTER 1: 정보의 가치",
    "CHAPTER 2: 정보 시스템의 개요",
    "CHAPTER 3: 정보평가",
    "CHAPTER 4: 정보를 통한 전략적 가치 창출",
    "CHAPTER 5: 정보 저장 및 조직화",
    "CHAPTER 6: 경영 의사결정을 위한 정보 분석",
    "CHAPTER 7: 정보 전송",
    "CHAPTER 8: 정보 보안",
    "CHAPTER 9: 기밀 유지 및 정보 프라이버시",
    "CHAPTER 10: 정보 시스템 개발",
    "CHAPTER 11: 정보 기반 비즈니스 프로세스",
    "CHAPTER 12: 전사적 정보 시스템",
    "CHAPTER 13: e–비즈니스의 정보",
    "CHAPTER 14: 경영 의사결정을 위한 정보와 지식",
];

// --- Mock Data ---
const mockTeacherData = {
  lowEngagementTopics: [
    { topic: "전문가 시스템 (Expert Systems)", percentage: 65 },
    { topic: "관계형 데이터베이스 정규화", percentage: 52 },
  ],
  lowEngagementTimes: [
    { time: "학습 후반 (40분 이후)", percentage: 70 },
  ],
  llmSolution: "학생들이 '전문가 시스템'과 같은 추상적인 개념을 어려워하는 경향이 있습니다."
};

// --- UI 컴포넌트들 ---
const DashboardCard = ({ title, children, className }) => (
  <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
    <h3 className="text-xl font-bold text-gray-800 mb-4">{title}</h3>
    <div className="space-y-2">{children}</div>
  </div>
);

const ChatWindow = ({ messages, onSendMessage }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

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
                            <div className="prose text-sm" style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            <div className="mt-4 flex gap-2">
                <input 
                    type="text" 
                    placeholder="메시지를 입력하세요..." 
                    className="flex-grow p-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                />
                <button 
                    onClick={handleSend}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
                >전송</button>
            </div>
        </div>
    );
};

// --- 로그인 화면 컴포넌트 (새로 추가) ---
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

// --- 학생 뷰 (이전 코드 기반 + 새 UI 적용) ---
const StudentView = () => {
  // 새로 추가: 로그인 상태 관리
  const [phase, setPhase] = useState('login');
  const [studentInfo, setStudentInfo] = useState(null);
  
  // 기존 상태들 (이전 코드 유지)
  const [messages, setMessages] = useState([]);
  const socketRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  const [realtimeFeedback, setRealtimeFeedback] = useState({ concentration: 'N/A', noise: 'N/A', start_time: '0', end_time: '0' });
  const [finalReport, setFinalReport] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const currentRecorderRef = useRef(null);
  const nextRecorderRef = useRef(null);
  const cleanupTimeoutsRef = useRef(null);

  // 새로 추가: 로그인 핸들러
  const handleLogin = (id, topic) => {
    setStudentInfo({ id, topic });
    setPhase('camera_setup');
  };

  // WebSocket 연결 (이전 코드 그대로 유지)
  useEffect(() => {
    console.log("🔄 WebSocket 연결 시도 중...");
    const ws = new WebSocket(API_URL);
    
    ws.onopen = () => {
        console.log("✅ WebSocket 연결 성공");
        socketRef.current = ws;
        setError(null);
    };
    
    ws.onclose = (event) => {
        console.log("🔒 WebSocket 연결 종료");
    };
    
    ws.onerror = (err) => {
        console.error("❌ WebSocket 오류:", err);
        setError("서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.");
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log("백엔드 메시지 수신:", message);
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
          console.log('최종 리포트 수신 완료. 웹소켓 연결을 종료합니다.');
          if (socketRef.current) {
            socketRef.current.close();
          }
          break;
        case 'chat_response':
          // AI Teacher 응답 처리
          const aiMessage = { sender: 'llm', text: message.message };
          setMessages(prevMessages => [...prevMessages, aiMessage]);
          break;
        case 'error':
          setError(message.message);
          break;
      }
    };

    return () => {
      if (ws) ws.close();
    };
  }, []);

  // 카메라 스트림 설정 (phase 기반으로 수정)
  useEffect(() => {
    if (phase === 'camera_setup') {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          mediaStreamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.style.transform = 'scaleX(-1)';
          }
        })
        .catch(err => {
          console.error("카메라 접근 오류:", err);
          setError("카메라/마이크 권한이 필요합니다. 권한을 허용하고 페이지를 새로고침해주세요.");
        });
    } else {
      if (mediaStreamRef.current) {
        const tracks = mediaStreamRef.current.getTracks();
        tracks.forEach(track => track.stop());
        mediaStreamRef.current = null;
      }
    }
  }, [phase]);

  // 오디오 처리 함수 (이전 코드 그대로)
  const processAudioChunk = async (audioBlob) => {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    
    const wavBuffer = toWav(audioBuffer);
    const audio = btoa(String.fromCharCode(...new Uint8Array(wavBuffer)));

    let frame = "";
    if (videoRef.current && videoRef.current.readyState === 4) {
        const canvas = document.createElement('canvas');
        canvas.width = 1080;
        canvas.height = 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        frame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
            type: 'data_chunk',
            frame,
            audio
        }));
    }
  };

  // 스트리밍 시작 (수정: studentInfo 사용)
  const startStreaming = () => {
    const socket = socketRef.current;
    let recorderIndex = 0;
    let timeoutIds = [];
    
    const cleanup = () => {
        timeoutIds.forEach(id => clearTimeout(id));
        timeoutIds = [];
    };
    cleanupTimeoutsRef.current = cleanup;

    if (socket && socket.readyState === WebSocket.OPEN) {
      // 수정된 부분: studentInfo에서 데이터 가져오기
      socket.send(JSON.stringify({ 
        type: 'start_session', 
        user_name: studentInfo?.id || '', 
        topic: studentInfo?.topic || '' 
      }));

      // 나머지 로직은 이전 코드 그대로 유지
      if (mediaStreamRef.current) {
        const audioStream = new MediaStream(mediaStreamRef.current.getAudioTracks());

        const createRecorder = (index) => {
          const mediaRecorder = new window.MediaRecorder(audioStream, { mimeType: 'audio/webm' });
          mediaRecorder.ondataavailable = async (e) => {
            if (e.data.size > 0) {
              try{
                await processAudioChunk(e.data);
              } catch (error) {
                console.error(`Recorder ${index} 오류:`, error);
              }
            }
          };
          return mediaRecorder;
        };

        const startNextRecorder = () => {
          recorderIndex++;
          nextRecorderRef.current = createRecorder(recorderIndex);

          const timeoutId1 = setTimeout(() => {
              if (nextRecorderRef.current && nextRecorderRef.current.state === 'inactive') {
                  nextRecorderRef.current.start();
              }
          }, 900);
          timeoutIds.push(timeoutId1);
        
          const timeoutId2 = setTimeout(() => {
              if (currentRecorderRef.current && currentRecorderRef.current.state === 'recording') {
                  currentRecorderRef.current.stop();
              }
              currentRecorderRef.current = nextRecorderRef.current;
              nextRecorderRef.current = null;
              startNextRecorder();
          }, 1000);
          timeoutIds.push(timeoutId2);
        };
      
        currentRecorderRef.current = createRecorder(recorderIndex);
        currentRecorderRef.current.start();
        startNextRecorder();
      }
    }
  };

  // 스트리밍 중지 (이전 코드 그대로)
  const stopStreaming = () => {
    if (cleanupTimeoutsRef.current) {
        cleanupTimeoutsRef.current();
        cleanupTimeoutsRef.current = null;
    }

    if (currentRecorderRef.current && currentRecorderRef.current.state === 'recording') {
        currentRecorderRef.current.stop();
        currentRecorderRef.current = null;
    }
    if (nextRecorderRef.current && nextRecorderRef.current.state === 'recording') {
        nextRecorderRef.current.stop();
        nextRecorderRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'end_session' }));
    }

    setIsStreaming(false);
    setPhase('loading_feedback');
  };

  const setupChat = (mode) => {
    if (mode === 'test') {
      setMessages([{ sender: 'llm', text: "테스트 기능은 현재 개발 중입니다." }]);
      setPhase('test');
    } else if (mode === 'feedback_chat') {
      setMessages([{ 
        sender: 'llm', 
        text: `안녕하세요, ${studentInfo?.id}님! 저는 여러분의 학습을 도와드리는 AI 교사입니다. 방금 완료하신 '${studentInfo?.topic}' 학습에 대해 분석한 결과를 바탕으로 궁금한 점이나 어려웠던 부분에 대해 자세히 설명드릴 수 있습니다. 어떤 것이 궁금하신가요?` 
      }]);
      setPhase('feedback_chat');
    }
  };
  
  const handleSendMessage = async (userMessage) => {
  const newMessages = [...messages, { sender: 'user', text: userMessage }];
  setMessages(newMessages);
  
  if (sessionId) {
    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage,
          user_name: studentInfo?.id || '학생',
          topic: studentInfo?.topic || '경영정보시스템'
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        const aiMessage = { sender: 'llm', text: data.response };
        setMessages(prevMessages => [...prevMessages, aiMessage]);
      } else {
        const errorMessage = { 
          sender: 'llm', 
          text: "답변 생성에 문제가 발생했습니다. 다시 시도해주세요." 
        };
        setMessages(prevMessages => [...prevMessages, errorMessage]);
      }
      
    } catch (error) {
      console.error('채팅 API 호출 오류:', error);
      const fallbackMessage = { 
        sender: 'llm', 
        text: "네트워크 연결에 문제가 있습니다. 다시 시도해주세요." 
      };
      setMessages(prevMessages => [...prevMessages, fallbackMessage]);
    }
  } else {
    const fallbackResponse = { 
      sender: 'llm', 
      text: "세션 정보가 없습니다. 학습을 다시 진행해주세요." 
    };
    setMessages([...newMessages, fallbackResponse]);
  }
};

  if (error) {
      return <div className="text-center p-8 bg-red-100 text-red-700 rounded-xl shadow-lg w-full"><b>오류:</b> {error}</div>
  }

  // phase별 렌더링
  switch (phase) {
    case 'login':
      return <LoginScreen onLogin={handleLogin} />;

    case 'camera_setup':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg w-full">
          <div className="text-left mb-4 bg-gray-100 p-3 rounded-lg">
            <p className="text-sm text-gray-600"><strong>회원번호:</strong> {studentInfo?.id}</p>
            <p className="text-sm text-gray-600"><strong>강의주제:</strong> {studentInfo?.topic}</p>
          </div>
          
          <h2 className="text-3xl font-bold text-gray-800 mb-4">학습 준비</h2>
          <p className="text-gray-600 mb-6">카메라가 켜지면 학습을 시작해주세요.</p>
          
          <div className="relative w-full">
            <video ref={videoRef} autoPlay muted className="w-full bg-black aspect-video rounded-lg mb-6 transform -scale-x-100"></video>
            {isStreaming && 
                <div className="absolute bottom-4 left-4 bg-gray-900 bg-opacity-75 text-white text-base p-4 rounded-xl shadow-lg">
                    <p className="font-bold">집중도 체크 시간: <span className="font-normal">{realtimeFeedback.start_time}초 ~ {realtimeFeedback.end_time}초</span></p>
                    <p className="font-bold">실시간 집중도: <span className="font-normal">{realtimeFeedback.concentration}</span></p>
                    <p className="font-bold">주변 소음: <span className="font-normal">{realtimeFeedback.noise}</span></p>
                </div>
            }
          </div>
          
          {!isStreaming ? (
            <button
                onClick={startStreaming}
                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-lg text-lg transition-transform transform hover:scale-105"
            > 학습 시작 </button>
          ) : (
            <button
                onClick={stopStreaming}
                className="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-8 rounded-lg text-lg transition-transform transform hover:scale-105"
            > 학습 종료 </button>
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
          <div className="flex justify-center gap-4">
            <button onClick={() => setupChat('test')} className="bg-green-500 hover:bg-green-600 text-white font-bold py-4 px-8 rounded-lg text-lg transition-transform transform hover:scale-105">
              📝 테스트 보기
            </button>
            <button onClick={() => setPhase('feedback_summary')} className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-4 px-8 rounded-lg text-lg transition-transform transform hover:scale-105">
              📊 피드백 보기
            </button>
          </div>
        </div>
      );

    case 'feedback_summary':
        return (
            <div className="p-8 bg-white rounded-xl shadow-lg w-full animate-fade-in">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-3xl font-bold text-gray-800">학습 피드백 리포트</h2>
                    <button
                        onClick={() => setPhase('choice')}
                        className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg"
                    >
                        ← 이전으로
                    </button>
                </div>
                {finalReport?.llm_report && 
                    <DashboardCard title="🤖 AI 종합 피드백" className="mt-6">
                        <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: finalReport.llm_report }} />
                    </DashboardCard>
                }
                <div className="text-center mt-8">
                    <button onClick={() => setupChat('feedback_chat')} className="bg-gray-700 hover:bg-gray-800 text-white font-bold py-3 px-8 rounded-lg text-lg transition-transform transform hover:scale-105">
                        💬 AI와 대화하며 더 알아보기
                    </button>
                </div>
            </div>
        );

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

// 교사 뷰 (기존 그대로)
const TeacherView = () => {
  const { lowEngagementTopics, lowEngagementTimes, llmSolution } = mockTeacherData;
  return (
    <div className="w-full max-w-4xl space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 text-center">교사용 대시보드</h2>
      <div className="grid md:grid-cols-2 gap-6">
        <DashboardCard title="📊 집중도 저하 주요 토픽">
          {lowEngagementTopics.map(item => (
            <div key={item.topic}>
              <div className="flex justify-between text-sm font-medium text-gray-600">
                <span>{item.topic}</span>
                <span>{item.percentage}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div className="bg-orange-500 h-2.5 rounded-full" style={{ width: `${item.percentage}%` }}></div>
              </div>
            </div>
          ))}
        </DashboardCard>
        <DashboardCard title="⏰ 집중력 저하 시간대">
          {lowEngagementTimes.map(item => (
             <div key={item.time}>
              <div className="flex justify-between text-sm font-medium text-gray-600">
                <span>{item.time}</span>
                <span>{item.percentage}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div className="bg-indigo-500 h-2.5 rounded-full" style={{ width: `${item.percentage}%` }}></div>
              </div>
            </div>
          ))}
        </DashboardCard>
      </div>
      <DashboardCard title="💡 교육 솔루션 제안 (LLM 기반)">
        <p className="text-gray-700 leading-relaxed">{llmSolution}</p>
      </DashboardCard>
    </div>
  );
};

// 메인 앱
function App() {
  const [view, setView] = useState('student');

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