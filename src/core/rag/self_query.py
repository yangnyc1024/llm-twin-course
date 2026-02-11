import opik  # 可观测性/追踪：用于函数级别 trace 记录
from config import settings  # 项目配置：读取 OpenAI 模型 ID、API Key 等
from langchain_openai import ChatOpenAI  # LangChain 封装的 OpenAI Chat 模型
from opik.integrations.langchain import OpikTracer  # Opik 与 LangChain 集成的 tracer

import core.logger_utils as logger_utils  # 日志工具：结构化日志记录
from core import lib  # 通用工具库：这里用于拆分用户姓名
from core.db.documents import UserDocument  # 用户文档模型：用于查找或创建用户记录
from core.rag.prompt_templates import SelfQueryTemplate  # SelfQuery 专用 Prompt 模板

logger = logger_utils.get_logger(__name__)  # 初始化当前模块 logger


class SelfQuery:
    opik_tracer = OpikTracer(tags=["SelfQuery"])
    # 定义 Opik tracer：为该类的链路添加标签，便于在观测平台中分类查看

    @staticmethod
    @opik.track(name="SelQuery.generate_response")
    # 使用 opik.track 对该方法进行函数级别追踪（记录输入输出、耗时等）
    def generate_response(query: str) -> str | None:
        prompt = SelfQueryTemplate().create_template()
        # 构造 Prompt 模板：通常用于从自然语言 query 中抽取结构化信息（如用户姓名）

        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        # 初始化 LLM：
        # temperature=0 表示尽量确定性输出（抽取任务通常希望稳定、可重复）

        chain = prompt | model
        # 构建 LangChain 执行链：prompt 输出直接传入 model

        chain = chain.with_config({"callbacks": [SelfQuery.opik_tracer]})
        # 为当前 chain 添加 Opik 回调 tracer：
        # 可以在每次 LLM 调用时记录 prompt / response / token 使用等信息

        response = chain.invoke({"question": query})
        # 调用模型：
        # question = 原始用户输入 query
        # 期望模型输出结构化的用户全名（或 none）

        response = response.content
        # 提取 LLM 返回的文本内容（LangChain Response 对象中 content 为核心文本）

        user_full_name = response.strip("\n ")
        # 清理输出：去除首尾换行和空格，避免格式问题影响后续逻辑

        if user_full_name == "none":
            return None
        # 如果模型明确返回 "none"，表示未识别到用户姓名，返回 None

        logger.info(
            f"Successfully extracted the user full name from the query.",
            user_full_name=user_full_name,
        )
        # 记录成功抽取到的用户全名（结构化字段便于日志分析）

        first_name, last_name = lib.split_user_full_name(user_full_name)
        # 使用工具函数拆分全名为 first_name / last_name
        # 通常假设格式为 "First Last"

        logger.info(
            f"Successfully extracted the user first and last name from the query.",
            first_name=first_name,
            last_name=last_name,
        )
        # 记录拆分结果，方便排查姓名解析问题

        user_id = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        # 在数据库中查找该用户：
        # 如果存在则返回已有 user_id
        # 如果不存在则创建新记录并返回其 ID

        return user_id  # 返回用户 ID（供上游用于多租户过滤或权限控制）
