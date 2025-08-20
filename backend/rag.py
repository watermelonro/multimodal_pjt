import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from typing import List, Dict, Optional, Tuple
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime
import ast

load_dotenv()


class RAGPipeline:
    def __init__(
        self, vector_db_path: str = "data/vector_store/경영정보시스템5_chroma_db"
    ):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # LLM 및 임베딩 초기화
        self.llm = ChatUpstage(
            api_key=self.api_key,  # type: ignore
            model="solar-pro2",
            temperature=0.1,
        )

        self.embeddings = UpstageEmbeddings(
            api_key=self.api_key,  # type: ignore
            model="embedding-query",
        )

        # 벡터스토어 로드
        self.vector_store = self._load_vector_store(vector_db_path)

        # RAG 체인 구성
        self.rag_chain = self._create_rag_chain()

        print(f"벡터 DB: {vector_db_path}")

    def _load_vector_store(self, vector_db_path: str) -> Chroma:
        """기존 벡터스토어 로드"""
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"벡터 DB를 찾을 수 없습니다: {vector_db_path}")

        print(f" 벡터스토어 로드 중: {vector_db_path}")

        vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings,
            collection_name="mis_textbook",
        )

        # 벡터스토어 상태 확인
        collection = vector_store._collection
        count = collection.count()
        print(f" 벡터스토어 로드 완료: {count}개 문서")

        return vector_store

    def _create_rag_chain(self):
        """RAG 체인 생성"""

        # 프롬프트 템플릿 정의 
        template = """You are an {topic} specialist. Give me MOST RELATIVE, ACCURATE TOPIC WORD based on textbook content provided.

**GUIDELINES:**
1. 교재 내용에 기반하여 정확한 대주제어를 제공하세요.
2. 대주제어가 여러 개이면 모두 적어주세요(최대 3개).
3. 각 주제어에는 교재 페이지 정보를 명시하세요.
4. 아래 출력 포멧을 꼭 지켜주세요.
5. 교재에 없는 내용이라면 추측하지 말고 "[["",""]]"를 출력하세요

**OUTPUT FORMAT:**
"[['주제어', '교재 페이지'], ['주제어2', '교재 페이지2'], ...]"
- 이 이외의 근거 같은 다른 내용은 절대 출력하지 말 것

**TEXTBOOK CONTENT:**
{context}

**주제어:**"""

        prompt = ChatPromptTemplate.from_template(template)

        # RAG 체인 구성 (간단하게 수정)
        chain = prompt | self.llm | StrOutputParser()

        return chain

    def _format_docs(self, docs) -> str:
        """검색된 문서들을 컨텍스트 형태로 포맷팅"""
        formatted_docs = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            content = doc.page_content

            # 너무 길면 자르기
            if len(content) > 500:
                content = content[:500] + "..."

            formatted_doc = f"""
[문서 {i}]
파일: {metadata.get("pdf_filename", "N/A")}
페이지: {metadata.get("page_number", "N/A")}
내용: {content}
"""
            formatted_docs.append(formatted_doc)

        return "\n".join(formatted_docs)

    def search_documents(
        self,
        query: str,
        k: int = 5,
        pdf_filter: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List:
        """고급 검색 기능"""

        # 기본 유사도 검색
        if pdf_filter or page_range:
            # 메타데이터 필터링이 필요한 경우
            all_docs = self.vector_store.similarity_search(
                query, k=k * 3
            )  # 더 많이 가져와서 필터링

            filtered_docs = []
            for doc in all_docs:
                metadata = doc.metadata

                # PDF 파일 필터
                if pdf_filter and pdf_filter not in metadata.get("pdf_filename", ""):
                    continue

                # 페이지 범위 필터
                if page_range:
                    page_num = metadata.get("page_number", 0)
                    if not (page_range[0] <= page_num <= page_range[1]):
                        continue

                filtered_docs.append(doc)

                if len(filtered_docs) >= k:
                    break

            docs = filtered_docs
        else:
            docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def ask_question(
        self,
        stt_input: str,
        topic: str,
        k: int = 5,
        pdf_filter: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
        show_sources: bool = False,
    ) -> Dict:
        """질문 답변 (메인 기능)"""

        # 1. 관련 문서 검색
        relevant_docs = self.search_documents(
            query=stt_input, k=k, pdf_filter=pdf_filter, page_range=page_range
        )

        if not relevant_docs:
            return {
                "query": stt_input,
                "relevant_docs": None,
                "answer": "죄송합니다. 해당 질문과 관련된 내용을 교재에서 찾을 수 없습니다.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
            }

        # 2. LLM으로 답변 생성
        print("🤖 RAG LLM 답변 생성 중...")
        try:
            # 문서들을 컨텍스트로 포맷팅
            formatted_context = self._format_docs(relevant_docs)

            # RAG 체인 실행
            answer = self.rag_chain.invoke(
                {
                    "topic": topic,
                    "context": formatted_context,
                }
            )
        except Exception as e:
            print(f" 답변 생성 실패: {str(e)}")
            return {
                "query": stt_input,
                "relevant_docs": relevant_docs,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
            }

        # 3. 소스 정보 수집
        sources = []
        for doc in relevant_docs:
            metadata = doc.metadata
            sources.append(
                {
                    "file": metadata.get("pdf_filename", "N/A"),
                    "page": metadata.get("page_number", "N/A"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "has_tables": metadata.get("has_tables", False),
                    "has_images": metadata.get("has_images", False),
                    "content_preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
            )

        # 4. 결과 출력
        answer = ast.literal_eval(answer)

        if show_sources:
            print("\n📚 참고 자료:")
            for i, source in enumerate(sources, 1):
                print(f"   [{i}] {source['file']} (p.{source['page']})")
                if source["has_tables"]:
                    print("       📋 표 포함")
                if source["has_images"]:
                    print("       🖼️ 이미지 포함")

        return {
            "query": stt_input,
            "relevant_docs": relevant_docs,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }

    def interactive_chat(self):
        """대화형 RAG 시스템"""
        topic = str(input("주제를 입력하세요:"))

        conversation_history = []

        while True:
            try:
                user_input = input("\n 내용: ").strip()

                if user_input.lower() in ["quit", "exit", "종료"]:
                    print("👋 RAG 시스템을 종료합니다.")
                    break

                elif user_input == "/help":
                    self._show_help()
                    continue

                elif user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                elif not user_input:
                    continue

                # 질문 처리
                result = self.ask_question(user_input, topic)
                conversation_history.append(result)

            except KeyboardInterrupt:
                print("\n👋 RAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f" 오류 발생: {str(e)}")

    def _show_help(self):
        """도움말 표시"""
        help_text = """
🔧 사용 가능한 명령어:
   /search <검색어>     - 단순 검색 (답변 생성 없이 문서만 찾기)
   /stats              - 벡터스토어 통계 정보
   /help               - 이 도움말
   quit/exit           - 프로그램 종료

💡 질문 예시:
   "경영정보시스템이란 무엇인가요?"
   "시스템의 구성요소는 무엇인가요?"
   "정보 처리 주기에 대해 설명해주세요"
        """
        print(help_text)

    def _handle_command(self, command: str):
        """명령어 처리"""
        if command.startswith("/search "):
            query = command[8:].strip()
            docs = self.search_documents(query)
            print(f"\n🔍 '{query}' 검색 결과:")
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                print(
                    f"   [{i}] {metadata.get('pdf_filename')} (p.{metadata.get('page_number')})"
                )

        elif command == "/stats":
            collection = self.vector_store._collection
            count = collection.count()
            print("\n 벡터스토어 통계:")
            print(f"    총 문서 수: {count}")
            print("    컬렉션명: mis_textbook")

        else:
            print(" 알 수 없는 명령어입니다. '/help'로 도움말을 확인하세요.")


if __name__ == "__main__":
    topic = str(input("topic:"))
    stt_input = str(input("content:"))
    RAGPipeline().ask_question(stt_input, topic)
