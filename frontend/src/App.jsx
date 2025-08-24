import React, { useState, useEffect, useRef } from 'react';

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
                <input type="text" placeholder="메시지를 입력하세요..." className="flex-grow p-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} />
                <button onClick={handleSend} className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors">전송</button>
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


// --- 페이지 컴포넌트 ---

const StudentView = () => {
  const [phase, setPhase] = useState('login');
  const [studentInfo, setStudentInfo] = useState(null);
  const [messages, setMessages] = useState([]);
  const videoRef = useRef(null);
  
  const mediaRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState(null);
  
  const [isCameraReady, setIsCameraReady] = useState(false);

  useEffect(() => {
    if (phase === 'camera_setup' || phase === 'recording') {
      setIsCameraReady(false); 
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            setIsCameraReady(true);
          }
        })
        .catch(err => {
            console.error("카메라 접근 오류:", err);
            setIsCameraReady(false);
        });
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    }
  }, [phase]);
  
  const handleStartRecording = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
      const chunks = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const videoUrl = URL.createObjectURL(blob);
        setRecordedVideoUrl(videoUrl);
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setPhase('recording');
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setPhase('choice');
    }
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

  // phase 상태에 따라 적절한 화면을 보여줍니다.
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
          <video ref={videoRef} autoPlay muted className="w-full bg-black aspect-video rounded-lg mb-6 transform -scale-x-100"></video>
          {!isRecording ? (
            <button 
              onClick={handleStartRecording} 
              disabled={!isCameraReady}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-lg text-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isCameraReady ? '학습 시작 및 녹화' : '카메라 준비 중...'}
            </button>
          ) : (
            <button onClick={handleStopRecording} className="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-8 rounded-lg text-lg">
              학습 종료
            </button>
          )}
        </div>
      );
    
    case 'choice':
      return (
        <div className="text-center p-8 bg-white rounded-xl shadow-lg animate-fade-in">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">학습 세션 완료!</h2>
          <p className="text-gray-600 mb-8">수고하셨습니다! 다음 단계를 선택해주세요.</p>
          {recordedVideoUrl && (
            <div className="mb-6">
              <a href={recordedVideoUrl} download={`${studentInfo.id}_${studentInfo.topic}.webm`} className="text-blue-600 hover:underline">
                녹화된 학습 영상 다운로드
              </a>
            </div>
          )}
          <div className="flex justify-center gap-4">
            <button onClick={() => setupChat('test')} className="bg-green-500 hover:bg-green-600 text-white font-bold py-4 px-8 rounded-lg text-lg">
              📝 테스트 보기
            </button>
            <button onClick={() => setPhase('feedback_summary')} className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-4 px-8 rounded-lg text-lg">
              📊 피드백 보기
            </button>
          </div>
        </div>
      );
      
    case 'feedback_summary':
        return (
            <div className="p-8 bg-white rounded-xl shadow-lg w-full animate-fade-in">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-3xl font-bold text-gray-800">학습 피드백 요약</h2>
                    {/* --- 추가된 부분: 이전 버튼 --- */}
                    <button 
                        onClick={() => setPhase('choice')} 
                        className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg"
                    >
                        ← 이전으로
                    </button>
                </div>
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
                <div className="text-center">
                    <button onClick={() => setupChat('feedback_chat')} className="bg-gray-700 hover:bg-gray-800 text-white font-bold py-3 px-8 rounded-lg text-lg">
                        💬 AI와 대화하며 더 알아보기
                    </button>
                </div>
            </div>
        );
        
    case 'test':
    case 'feedback_chat':
      return (
        <div className="w-full">
            {/* --- 추가된 부분: 이전 버튼 --- */}
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
