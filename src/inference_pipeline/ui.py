import sys
from pathlib import Path

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
# 为了模拟多模块（例如 core 与 feature_pipeline）的项目结构，
# 手动将 ./src 目录添加到 Python 的模块搜索路径中。
# ⚠️ 注意：仅在开发或教学环境中使用，生产环境请使用标准包导入结构。
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core.config import settings
from llm_twin import LLMTwin

# 修改配置以支持本地环境（如替换服务地址为 localhost）
settings.patch_localhost()


import gradio as gr
from inference_pipeline.llm_twin import LLMTwin

# 初始化 LLM Twin 模型接口对象（mock=False 表示实际调用模型）
llm_twin = LLMTwin(mock=False)


def predict(message: str, history: list[list[str]], author: str) -> str:
    """
    Generates a response using the LLM Twin, simulating a conversation with your digital twin.

    Args:
        message (str): The user's input message or question.
        history (List[List[str]]): Previous conversation history between user and twin.
        about_me (str): Personal context about the user to help personalize responses.

    Returns:
        str: The LLM Twin's generated response.
    """
    # 说明：
    # 该函数是 Gradio 的回调函数，用于处理每一次用户输入。
    # 它将用户输入封装为 query，并调用 llm_twin.generate() 进行推理。

    # 构造输入查询，将用户身份与输入合并，增强个性化生成
    query = f"I am {author}. Write about: {message}"

    # 调用 LLM Twin 的生成方法
    response = llm_twin.generate(
        query=query, enable_rag=True, sample_for_evaluation=False
    )

    # 返回模型生成的回答文本
    return response["answer"]


# 使用 Gradio 构建聊天界面
demo = gr.ChatInterface(
    predict,  # 调用的预测函数
    textbox=gr.Textbox(
        placeholder="Chat with your LLM Twin",  # 输入框提示文字
        label="Message",  # 标签
        container=False,
        scale=7,  # 控件占比
    ),
    additional_inputs=[
        # 额外输入项（用户身份）
        gr.Textbox(
            "Paul Iusztin",
            label="Who are you?",  # 标签提示用户输入自己的身份
        )
    ],
    title="Your LLM Twin",  # 页面标题
    description="""
    Chat with your personalized LLM Twin! This AI assistant will help you write content incorporating your style and voice.
    """,  # 描述文字
    theme="soft",  # 使用柔和主题
    examples=[
        # 示例输入，用于演示或快速测试
        [
            "Draft a post about RAG systems.",
            "Paul Iusztin",
        ],
        [
            "Draft an article paragraph about vector databases.",
            "Paul Iusztin",
        ],
        [
            "Draft a post about LLM chatbots.",
            "Paul Iusztin",
        ],
    ],
    cache_examples=False,  # 禁用示例缓存
)


if __name__ == "__main__":
    # 启动 Gradio Web 应用，允许外部访问（0.0.0.0）并共享访问链接
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
