import json

from config import settings
from core import get_logger
from openai import OpenAI

# -------------------------------------------
# ✅ 模型参数与日志配置
# -------------------------------------------
MAX_LENGTH = 16384  # 限制 prompt 最长长度（避免超过 OpenAI 模型输入上限）
SYSTEM_PROMPT = (
    "You are a technical writer handing someone's account to post about AI and MLOps."
)
# 系统提示词（system role），告诉模型“你是一个擅长写技术内容的撰稿人”
# 用于引导模型以专业语气生成 instruction。

logger = get_logger(__name__)


# ======================================================
# 🧠 GptCommunicator：GPT API 通信封装类
# ======================================================
class GptCommunicator:
    """
    GptCommunicator 类封装了与 OpenAI GPT API 通信的核心逻辑。

    功能：
    - 根据给定的 prompt，调用 GPT 模型（如 gpt-4-turbo / gpt-3.5）；
    - 获取模型输出；
    - 清洗、解析响应为 JSON 对象；
    - 捕获并记录异常，避免中断整个批次。

    用途：
    - 在 DatasetGenerator 中被调用；
    - 用于根据输入内容自动生成“instruction + content” 数据对。
    """

    def __init__(self, gpt_model: str = settings.OPENAI_MODEL_ID):
        # 从配置文件读取 API Key 与模型名称
        self.api_key = settings.OPENAI_API_KEY
        self.gpt_model = gpt_model

    # ------------------------------------------------------
    # 核心方法：发送 prompt 到 GPT，并返回解析后的结果
    # ------------------------------------------------------
    def send_prompt(self, prompt: str) -> list:
        """
        向 OpenAI Chat API 发送 prompt，并解析返回的 JSON 格式响应。

        参数：
            prompt (str): 经过格式化后的 prompt（由 DataFormatter 构造）

        返回：
            list: 从 GPT 返回的 JSON 结果解析出的 Python 列表对象。
                  每个元素通常是 {"instruction": "...", "content": "..."}。
        """
        try:
            # 初始化 OpenAI 客户端
            client = OpenAI(api_key=self.api_key)
            logger.info(f"Sending batch to GPT = '{settings.OPENAI_MODEL_ID}'.")

            # 调用 ChatCompletion 接口
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示词
                    {"role": "user", "content": prompt[:MAX_LENGTH]},  # 用户输入（截断到安全长度）
                ],
                model=self.gpt_model,
            )

            # 提取模型回复的文本内容
            response = chat_completion.choices[0].message.content

            # 调用 clean_response() 去掉多余字符，只保留有效 JSON
            return json.loads(self.clean_response(response))

        except Exception:
            # 捕获所有异常（如网络错误、API超时、解析失败等）
            logger.exception(
                f"Skipping batch! An error occurred while communicating with API."
            )
            return []  # 返回空列表以保证 pipeline 不中断

    # ------------------------------------------------------
    # 辅助方法：清洗 GPT 返回的字符串，使其变为合法 JSON
    # ------------------------------------------------------
    @staticmethod
    def clean_response(response: str) -> str:
        """
        从 GPT 返回的文本中提取有效 JSON 字符串。

        原因：
        - GPT 有时会在返回 JSON 前后加上说明文字或 Markdown；
        - 此函数通过找到第一个 '[' 和最后一个 ']' 来截取真正的 JSON 区段。

        参数：
            response (str): GPT 原始响应字符串

        返回：
            str: 清洗后的 JSON 格式字符串
        """
        start_index = response.find("[")  # 找到 JSON 列表起始位置
        end_index = response.rfind("]")   # 找到 JSON 列表结束位置
        return response[start_index : end_index + 1]



# DatasetGenerator
#     ↓
# DataFormatter.format_prompt()  →  生成一个长 prompt（包含多条内容）
#     ↓
# GptCommunicator.send_prompt(prompt)
#     ↓
# OpenAI GPT 模型生成输出（含 instruction + content）
#     ↓
# GptCommunicator.clean_response() 清洗响应
#     ↓
# 返回 JSON 对象列表 → DatasetGenerator 整合成数据集
