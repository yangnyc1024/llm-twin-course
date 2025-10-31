import json
from typing import Any

from config import settings
from opik.evaluation.metrics import base_metric, exceptions, score_result
from opik.evaluation.models import litellm_chat_model
from pydantic import BaseModel


class LLMJudgeStyleOutputResult(BaseModel):
    """
    定义 LLM 评估结果的数据结构：
    - score：模型评分（数值）
    - reason：评分原因说明（字符串）
    """
    score: int
    reason: str


class Style(base_metric.BaseMetric):
    """
    自定义指标：用于评估 LLM 输出的语气和写作风格是否适合博客或社交媒体内容。

    原理：
    - 使用一个辅助的 LLM（即“评判模型”）来对主模型的输出进行风格判定。
    - 如果风格合适则返回高分，否则返回低分。
    
    评分标准：
    - 1.0：风格完全合适（博客/社交媒体可读）
    - 0.5：风格中等（部分正式或复杂表达）
    - 0.0：风格不合适（过于学术或正式）
    """

    def __init__(
        self, name: str = "style_metric", model_name: str = settings.OPENAI_MODEL_ID
    ) -> None:
        # 指标名称（默认 "style_metric"）
        self.name = name
        # 初始化一个轻量版的 LLM 客户端，用于生成评判结果
        self.llm_client = litellm_chat_model.LiteLLMChatModel(model_name=model_name)
        # 定义提示词模板（prompt template）
        # 用于指示评判模型如何分析输出风格，并生成评分与原因
        self.prompt_template = """
        You are an impartial expert judge. Evaluate the quality of a given answer to an instruction based on it's style. 
Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

Instruction: {input}

Answer: {output}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{
        "reason": "...",
        "score": 0
    }},
    "style": {{
        "reason": "...",
        "score": 0
    }}
}}
"""

    def score(self, input: str, output: str, **ignored_kwargs: Any):
        """
        调用评判 LLM，对主模型的输出进行打分。

        Args:
            input: 模型输入（指令）
            output: 模型生成的输出
            **ignored_kwargs: 额外参数（为与 evaluate 函数兼容而保留）

        流程：
        1. 生成评判用的 prompt；
        2. 调用 LLM 返回结构化评分；
        3. 解析评分并返回标准化结果。
        """

        # 将输入和输出填充到提示词模板中
        prompt = self.prompt_template.format(input=input, output=output)

        # 调用 LLM 客户端进行生成，并要求以 LLMJudgeStyleOutputResult 格式返回
        model_output = self.llm_client.generate_string(
            input=prompt, response_format=LLMJudgeStyleOutputResult
        )

        # 对模型输出进行解析，提取数值分数与理由
        return self._parse_model_output(model_output)

    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        """
        解析 LLM 评判模型返回的内容并构造标准化评分结果。

        步骤：
        1. 尝试将返回内容解析为 JSON；
        2. 提取 score 与 reason；
        3. 校验 score 是否在有效范围 [1, 3]；
        4. 将评分标准化为 [0, 1] 区间；
        5. 返回 ScoreResult 对象。
        """
        try:
            dict_content = json.loads(content)  # 尝试解析 JSON 格式输出
        except Exception:
            raise exceptions.MetricComputationError("Failed to parse the model output.")  # 若解析失败则抛出异常

        score = dict_content["score"]  # 提取分数
        try:
            assert 1 <= score <= 3, f"Invalid score value: {score}"  # 校验分数是否在 [1,3] 区间内
        except AssertionError as e:
            raise exceptions.MetricComputationError(str(e))  # 分数越界则抛异常

        score = (score - 1) / 2.0  # 将分数标准化到 [0,1] 区间

        # 返回结构化的评分结果（名称、值、原因）
        return score_result.ScoreResult(
            name=self.name,
            value=score,
            reason=dict_content["reason"],
        )
