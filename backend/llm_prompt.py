import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from rag import RAGPipeline

load_dotenv()


class LLMPipeline:
    def __init__(self):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되지 않았습니다.")

        self.llm = ChatUpstage(
            api_key=self.api_key,  # type: ignore
            model="solar-pro2",
            temperature=0.1,
        )

        self.rag = RAGPipeline()

        self.relevant_docs = []

        self.chain = self._create_chain()

    def _create_chain(self):
        """LLM 체인 생성"""
        template = """
당신은 20대 학생들에게 데이터를 바탕으로 학습 컨설팅을 해주는 전문 상담원입니다. 주 목표는 학생들이 {topic}의 학습 목표를 효과적으로 달성할 수 있도록 학습 집중도/몰입도와 관련된 피드백을 제공하는 것입니다.

**GUIDELINES:**
1. 20대 대학생들에게 적합한 말투, 친근한 말투를 사용할 것.
2. **없는 사실을 만들어내지 않고** 받은 데이터를 기반으로 하여 근거 위주의 상담을 진행할 것.
3. **확실하지 않는 사실은 추론하지 않을 것**
4. 받은 데이터의 모든 열을 충분히 활용하고 근거있는 설명을 만들기 위해 지속적으로 생각할 것.
5. 부족한 부분은 확실하게 짚어주고, 동기가 생길 수 있게 격려 위주로 상담할 것.
6. 각 피드백 당 **예시를 참고**하여 5-6문장 정도로 작성할 것.
7. 피드백을 **마크다운 형식**으로 써줘. 각 항목은 굵은 제목과 줄바꿈을 포함해서 구성해줘.

**DATA:**
{data}

**RAG:**
{rag}

**DATA INFO:**
- 'start_time': 학습 시작 시간
- 'concentration': 집중도 분류 예측
- 'pose_yaw': 좌우 시선(음수:오른쪽, 양수:왼쪽)
- 'pose_pitch': 상하 시선(음수:아래쪽, 양수:위쪽)
- 'noise': 잡음 예측 결과

**RAG INFO:**
- [[[[주제어, 교재 페이지], [주제어2, 교재 페이지2], ...], 집중여부], ...]

**피드백 1. "언제" 집중도가 높았는지/낮았는지에 대한 피드백**
- DATA의 'start_time'을 기준으로 시간에 따른 몰입도/집중도 피드백
- 'start_time'의 초단위를 시간/분 단위로 변환하여 설명
- 몰입도 예측 결과를 가장 중시하되, 시선 처리, 잡음 예측 결과 등을 부가적으로 종합하여 칭찬할 점과 개선할 점 언급
- **예시:** 너의 학습 패턴을 살펴본 결과, 학습 초반에는 잘 집중하는 것 같아. 아주 칭찬해^^ 다만 시간이 지남에 따라서 점차 집중도가 낮아진 모습이 보여. 특히 시선도 좌우로 조금 분산되는 것 같아. 또, 싸이렌 소리에 예민한 것 같아. 다음에 공부할 때는 이 점 신경써서 마지막까지 함께 집중해보도록 노력해보자~

**피드백 2. "어떤 학습 시에" 집중도가 높았는지/낮았는지에 대한 피드백**
- RAG를 기반으로 학습 주제에 따른 몰입도/집중도 피드백
- 집중도가 낮은 주제는 교재 페이지를 언급하면서 피드백 진행
- **예시:** 거래처리시스템(Transaction Processing Systems) 공부할 때는 엄청 집중하고 있었어. 정말 열심히 한다~ 반면 전문가 시스템(Expert Systems)에서 집중도가 많이 떨어졌었네… 이 부분은 교재의 ~~를 보면서 다시 체크해보면 좋을 것 같아!

**피드백 3. 최종정리**
- 과거 생성한 피드백을 참고할 것.
- 피드백 1과 피드백 2의 전반적인 흐름을 제대로 정확하게 알려줄 것
- **예시:** 공부하는 모습이나 시선, 주변 환경 등을 종합해보았을 때 지금 매우 잘하고 있다고 생각이 들어! 잡음이 있을 때도 집중하는 모습을 많이 보여 줬고, 시선도 강의에 집중을 잘하고 있었던 것 같아~ 아주 칭찬해^^ 하지만 몇 가지 같이 다듬어가면 좋을 부분은 시간에 따라 점차 집중도가 떨어지는 부분이야. 마지막까지 마음을 다잡고 같이 끝까지 해보자! 그리고 전문가 시스템을 이해하는 것에 어려움을 겪는 것 같아. 혹시 이 부분에 궁금한 점 있어? 있으면 같이 해결해나가보자!

**피드백 4. 추천 전략**
- 최종정리를 바탕으로 학습 전략 제시
- 과거 생성한 피드백을 참고할 것.

**출력 구조**
✅ 피드백 1. 학습 태도 피드백
...

☑️ 피드백 2. 학습 내용 피드백
...

📚 최종정리
...

💡 추천전략
...
"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    def _rag_process(self, topic: str, stt_chunk: list) -> str:
        """
        RAG 진행 [OPTIMIZED]
        """
        result = []

        # 텍스트 그룹화
        focused_texts = " ".join(
            [text for text, state in stt_chunk if state == "집중" and text.strip()]
        )
        unfocused_texts = " ".join(
            [text for text, state in stt_chunk if state == "집중 못함" and text.strip()]
        )

        # 집중 구간 RAG
        if focused_texts:
            rag_output = self.rag.ask_question(focused_texts, topic)
            topic_output = rag_output["answer"]
            if topic_output and topic_output[0][0] != "":
                result.append([topic_output, "집중"])
            self.relevant_docs.extend(rag_output["relevant_docs"])

        # 비집중 구간 RAG
        if unfocused_texts:
            rag_output = self.rag.ask_question(unfocused_texts, topic)
            topic_output = rag_output["answer"]
            if topic_output and topic_output[0][0] != "":
                result.append([topic_output, "집중 못함"])
            self.relevant_docs.extend(rag_output["relevant_docs"])

        return str(result)

    def print_report(self, topic: str, data: str, stt_chunk: list):
        """
        리포트 생성
        """
        try:
            print("🚀 LLM 리포트 생성 중...")
            rag = self._rag_process(topic, stt_chunk)

            answer = self.chain.invoke({"topic": topic, "data": data, "rag": rag})

        except Exception as e:
            print(f"답변 생성 실패: {str(e)}")
            return None
        return answer
    
    def generate_chat_response(self, user_message: str, user_name: str, 
                              topic: str, analysis_results: list) -> str:
        """교재 기반 학습 도움 + 개인화"""
        # 채팅 전용 프롬프트
        chat_template = """당신은 {topic} 교재를 완벽히 숙지한 친근한 AI 튜터입니다. {user_name}님의 질문에 교재 내용을 바탕으로 정확하고 이해하기 쉽게 답변해주세요.


        **학습 분석 정보:**
        {rag_result}

        **질문 관련 교재 내용:**
        {rag_result}

        **개인화 참고사항:**
        {personalization_hint}

        **학생 질문:** {user_message}

        **답변 지침:**
        1. 교재 내용을 중심으로 정확한 답변 제공
        2. 20대 대학생에게 적합한 친근한 말투 사용
        3. 개인화 참고사항이 있다면 자연스럽게 언급
        4. 3-4문장으로 간결하고 명확하게 설명
        5. 필요시 교재 페이지 번호 언급

        **답변:**"""

        # 1. RAG 검색 (주요)
        rag_result = self.rag.ask_question(user_message, topic, k=3, show_sources=False)
        rag_text = str(rag_result.get("answer", "해당 내용을 교재에서 찾기 어렵네요."))
        
        # 2. 개인화 힌트 (보조)
        personalization_hint = self._get_personalization_hint(analysis_results, user_name)
        
        # 3. 응답 생성
        chat_prompt = ChatPromptTemplate.from_template(chat_template)
        chat_chain = chat_prompt | self.llm | StrOutputParser()
        
        try:
            response = chat_chain.invoke({
                "user_name": user_name,
                "topic": topic,
                "user_message": user_message,
                "rag_result": rag_text,
                "personalization_hint": personalization_hint
            })
            return response
        except Exception as e:
            return "답변 생성 중 오류가 발생했습니다. 다시 질문해주세요."

    def _get_personalization_hint(self, analysis_results, user_name):
        """질문과 관련된 개인화 힌트 생성"""
        if not analysis_results:
            return "개인화 정보 없음"
        
        # 집중도가 낮았던 구간이 많다면
        low_focus_count = sum(1 for r in analysis_results if r.get('result', {}).get('str') == '낮음')
        if low_focus_count > len(analysis_results) * 0.3:
            return f"{user_name}님이 이전 학습에서 일부 어려움을 겪었던 부분과 관련될 수 있음"
        
        return "전반적으로 잘 이해하고 있는 학습자"