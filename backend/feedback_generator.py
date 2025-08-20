from graph import plot_feedback_and_df
from llm_prompt import LLMPipeline
import markdown
import base64

llm_model = LLMPipeline()

class GenerateFeedback:
    def __init__(self):
        pass

    def generate(self, topic: str, name: str, data: list) -> str:
        """
        topic : str type topic info
        name : str type user name
        data : list type model inference data
        """
        result = sorted(data, key=lambda x: x["timestamp"]["start"])
        # 그래프 이미지 데이터를 직접 받습니다.
        data_str, stt_chunk, image_bytes = plot_feedback_and_df(result, name)

        # 이미지 데이터를 Base64로 인코딩
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Concentration Graph" style="width:100%; max-width:800px; display:block; margin: 0 auto;"/>'

        feedback = llm_model.print_report(topic, data_str, stt_chunk)
        
        assert feedback is not None, "LLM으로부터 피드백을 생성하지 못했습니다."
        
        # 이미지 HTML을 피드백 내용 앞에 추가
        full_feedback_markdown = f"{image_html}\n\n{feedback}"
        
        markdown_feedback = markdown.markdown(full_feedback_markdown)
        return markdown_feedback