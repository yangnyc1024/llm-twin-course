import pprint

import opik
import sagemaker
from config import settings
from core import logger_utils
from core.opik_utils import add_to_dataset_with_sampling
from core.rag.retriever import VectorRetriever
from langchain.prompts import PromptTemplate
from opik import opik_context
from prompt_templates import InferenceTemplate
from sagemaker.huggingface.model import HuggingFacePredictor
from utils import compute_num_tokens, truncate_text_to_max_tokens

logger = logger_utils.get_logger(__name__)


class LLMTwin:
    """
    LLMTwin 主推理类，用于封装整个 LLM 调用流程：
    包括 Prompt 构建、RAG 检索、SageMaker 模型推理与监控采样。
    """

    def __init__(self, mock: bool = False) -> None:
        # mock 模式用于本地测试，不实际调用模型推理服务
        self._mock = mock
        # 创建 SageMaker 模型推理端点对象
        self._llm_endpoint = self.build_sagemaker_predictor()
        # 构建 Prompt 模板实例
        self.prompt_template_builder = InferenceTemplate()

    def build_sagemaker_predictor(self) -> HuggingFacePredictor:
        """
        构建 AWS SageMaker 的 HuggingFace 推理端点对象。
        endpoint_name 来自配置文件 settings.DEPLOYMENT_ENDPOINT_NAME。
        """
        return HuggingFacePredictor(
            endpoint_name=settings.DEPLOYMENT_ENDPOINT_NAME,
            sagemaker_session=sagemaker.Session(),
        )

    @opik.track(name="inference_pipeline.generate")
    def generate(
        self,
        query: str,
        enable_rag: bool = False,
        sample_for_evaluation: bool = False,
    ) -> dict:
        """
        主推理方法：生成模型回答。

        参数：
        - query: 用户输入的问题。
        - enable_rag: 是否启用 RAG（检索增强生成）。
        - sample_for_evaluation: 是否进行采样保存以便后续评估。
        """

        # 从模板构建器生成系统提示词（system_prompt）与模板对象
        system_prompt, prompt_template = self.prompt_template_builder.create_template(
            enable_rag=enable_rag
        )
        # 模板变量字典
        prompt_template_variables = {"question": query}

        # 若启用 RAG，则进行向量检索与重排序
        if enable_rag is True:
            retriever = VectorRetriever(query=query)
            hits = retriever.retrieve_top_k(
                k=settings.TOP_K, to_expand_to_n_queries=settings.EXPAND_N_QUERY
            )
            # rerank：对检索结果进行重排序后取前 K 条上下文
            context = retriever.rerank(hits=hits, keep_top_k=settings.KEEP_TOP_K)
            prompt_template_variables["context"] = context
        else:
            context = None

        # 格式化 prompt（构造 system/user messages）
        messages, input_num_tokens = self.format_prompt(
            system_prompt, prompt_template, prompt_template_variables
        )

        logger.debug(f"Prompt: {pprint.pformat(messages)}")

        # 调用大模型推理接口
        answer = self.call_llm_service(messages=messages)
        logger.debug(f"Answer: {answer}")

        # 统计输入输出的 token 数量
        num_answer_tokens = compute_num_tokens(answer)
        opik_context.update_current_trace(
            tags=["rag"],
            metadata={
                "prompt_template": prompt_template.template,
                "prompt_template_variables": prompt_template_variables,
                "model_id": settings.MODEL_ID,
                "embedding_model_id": settings.EMBEDDING_MODEL_ID,
                "input_tokens": input_num_tokens,
                "answer_tokens": num_answer_tokens,
                "total_tokens": input_num_tokens + num_answer_tokens,
            },
        )

        # 返回回答结果与上下文
        answer = {"answer": answer, "context": context}

        # 若启用采样模式，则将输入与输出样本加入监控数据集
        if sample_for_evaluation is True:
            add_to_dataset_with_sampling(
                item={"input": {"query": query}, "expected_output": answer},
                dataset_name="LLMTwinMonitoringDataset",
            )

        return answer

    @opik.track(name="inference_pipeline.format_prompt")
    def format_prompt(
        self,
        system_prompt,
        prompt_template: PromptTemplate,
        prompt_template_variables: dict,
    ) -> tuple[list[dict[str, str]], int]:
        """
        格式化 prompt：
        - 将模板变量填充到 prompt 模板中
        - 计算 token 数量并截断超长输入
        - 生成符合 LLM 接口格式的 messages 列表
        """
        # 用模板变量替换占位符
        prompt = prompt_template.format(**prompt_template_variables)

        # 计算系统提示词 token 数量
        num_system_prompt_tokens = compute_num_tokens(system_prompt)

        # 截断用户输入文本，使总长度不超过模型最大输入限制
        prompt, prompt_num_tokens = truncate_text_to_max_tokens(
            prompt, max_tokens=settings.MAX_INPUT_TOKENS - num_system_prompt_tokens
        )

        total_input_tokens = num_system_prompt_tokens + prompt_num_tokens

        # 构造符合 OpenAI Chat 格式的消息结构
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return messages, total_input_tokens

    @opik.track(name="inference_pipeline.call_llm_service")
    def call_llm_service(self, messages: list[dict[str, str]]) -> str:
        """
        调用 LLM 推理服务（AWS SageMaker 端点）。

        参数：
        - messages: prompt 消息列表（system + user）
        返回：
        - answer: 模型生成的文本回答
        """
        if self._mock is True:
            logger.warning("Mocking LLM service call.")
            # 模拟返回结果，不调用实际端点
            return "Mocked answer."

        # 通过 SageMaker 推理端点调用 LLM 服务
        answer = self._llm_endpoint.predict(
            data={
                "messages": messages,
                "parameters": {
                    "max_new_tokens": settings.MAX_TOTAL_TOKENS
                    - settings.MAX_INPUT_TOKENS,  # 控制生成的 token 数量
                    "temperature": 0.01,  # 控制生成随机性
                    "top_p": 0.6,  # nucleus sampling 参数
                    "stop": ["<|eot_id|>"],  # 停止标志
                    "return_full_text": False,  # 只返回生成部分
                },
            }
        )
rtwa
        # 提取回答文本
        answer = answer["choices"][0]["message"]["content"].strip()

        return answer
