import React, { useState, useEffect, useRef } from 'react';

// [FIX] This import was causing the white screen error and has been removed.
// import ReactMarkdown from 'react-markdown'; 

const API_URL = 'ws://localhost:8000/ws/lecture-analysis';
const BACKEND_URL = 'http://localhost:8000';

// --- Mock Data (백엔드 연동 전 임시 데이터) ---
const mockTeacherData = {
  lowEngagementTopics: [
    { topic: "전문가 시스템 (Expert Systems)", percentage: 65 },
    { topic: "관계형 데이터베이스 정규화", percentage: 52 },
  ],
  lowEngagementTimes: [
    { time: "학습 후반 (40분 이후)", percentage: 70 },
  ],
  llmSolution: "학생들이 '전문가 시스템'과 같은 추상적인 개념을 어려워하는 경향이 있습니다. 시각 자료나 실제 사례를 활용한 비유를 통해 개념을 설명하고, 학습 후반부에는 5분 정도의 짧은 휴식이나 스트레칭을 통해 집중력을 환기시키는 '뽀모도로 기법'을 도입하는 것을 추천합니다."
};

// --- 재사용 가능한 UI 컴포넌트 ---
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

// --- 페이지 컴포넌트 ---
const StudentView = () => {
  const [phase, setPhase] = useState('camera_setup');
  const [messages, setMessages] = useState([]);
  const socketRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  const [realtimeFeedback, setRealtimeFeedback] = useState({ concentration: 'N/A', noise: 'N/A' });
  const [finalReport, setFinalReport] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const streamIntervalRef = useRef(null);

  // WebSocket 연결 및 메시지 핸들러 설정
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
          if (streamIntervalRef.current) {
            clearInterval(streamIntervalRef.current);
          }
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
        case 'error':
          setError(message.message);
          break;
      }
    };

    return () => {
      if (ws) {
        ws.close();
      }
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    };
  }, []);

  // 카메라 스트림 설정
  useEffect(() => {
    if (phase === 'camera_setup') {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          mediaStreamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
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

  const startStreaming = () => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'start_session', user_name: '학생A', topic: '경영정보시스템' }));
        
        streamIntervalRef.current = setInterval(() => {
            if (videoRef.current && videoRef.current.readyState === 4) {
                const canvas = document.createElement('canvas');
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                const frame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                
                const audio = ""; 

                if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                    socketRef.current.send(JSON.stringify({ type: 'data_chunk', frame, audio }));
                }
            }
        }, 1000 / 2); // Reduced frequency for testing
    }
  };

  const stopStreaming = () => {
    // 1. 데이터 전송 중지
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }

    // 2. 카메라 스트림 중지
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    // 3. 백엔드에 세션 종료 메시지 전송
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
        console.log('백엔드로 end_session 메시지를 전송합니다.');
        socket.send(JSON.stringify({ type: 'end_session' }));
    }

    // 4. UI 상태를 로딩으로 즉시 변경
    setIsStreaming(false);
    setPhase('loading_feedback');
  };
  
  const handleSendMessage = (userMessage) => {
    const newMessages = [...messages, { sender: 'user', text: userMessage }];
    setMessages(newMessages);
    const botResponse = { sender: 'llm', text: "죄송합니다. 채팅 기능은 현재 개발 중입니다." };
    setMessages([...newMessages, botResponse]);
  };

  if (error) {
      return <div className="text-center p-8 bg-red-100 text-red-700 rounded-xl shadow-lg w-full"><b>오류:</b> {error}</div>
  }

  switch (phase) {
    case 'camera_setup':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg w-full">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">학습 준비</h2>
          <p className="text-gray-600 mb-6">카메라가 켜지면 학습을 시작해주세요. 학습 중 여러분의 모습을 분석합니다.</p>
          <div className="relative w-full">
            <video ref={videoRef} autoPlay muted className="w-full bg-black aspect-video rounded-lg mb-6 transform -scale-x-100"></video>
            {isStreaming && 
                <div className="absolute bottom-4 left-4 bg-gray-900 bg-opacity-75 text-white text-base p-4 rounded-xl shadow-lg">
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
        if (!finalReport) return <div className="text-center p-8">피드백 데이터가 없습니다.</div>;
        return (
            <div className="p-8 bg-white rounded-xl shadow-lg w-full animate-fade-in">
                <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">학습 피드백 리포트</h2>
                {finalReport.llm_report && 
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
      return <ChatWindow messages={messages} onSendMessage={handleSendMessage} />;
    default:
      return null;
  }
};

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
      <div className="w-full max-w-4xl">
        {view === 'student' ? <StudentView /> : <TeacherView />}
      </div>
    </div>
  );
}

export default App;