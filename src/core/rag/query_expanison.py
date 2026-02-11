import opik  # 可观测性/追踪：用于函数级别 trace 记录
from config import settings  # 项目配置：读取 OpenAI 模型 ID 和 API Key
from langchain_openai import ChatOpenAI  # LangChain 封装的 OpenAI Chat 模型
from opik.integrations.langchain import OpikTracer  # Opik 与 LangChain 集成的 tracer

from core.rag.prompt_templates import QueryExpansionTemplate  # Query Expansion 专用 Prompt 模板


class QueryExpansion:
    opik_tracer = OpikTracer(tags=["QueryExpansion"])
    # 定义 Opik tracer：为该类相关调用打标签，方便在观测系统中区分模块

    @staticmethod
    @opik.track(name="QueryExpansion.generate_response")
    # 使用 opik.track 对方法进行函数级别追踪（记录调用链路、耗时、输入输出等）
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        # 实例化 Query Expansion 模板对象（内部定义分隔符和 prompt 结构）

        prompt = query_expansion_template.create_template(to_expand_to_n)
        # 根据目标扩展数量 to_expand_to_n 构建 prompt
        # 通常会在指令中要求模型生成 N 条语义相近但表达不同的查询

        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        # 初始化 LLM：
        # temperature=0 保证输出更稳定（扩展查询希望可控、格式一致）

        chain = prompt | model
        # 使用 LangChain LCEL 语法构建执行链：prompt 输出作为 model 输入

        chain = chain.with_config({"callbacks": [QueryExpansion.opik_tracer]})
        # 为该链路添加 Opik 回调：
        # 可以追踪 prompt、响应、token 使用等信息

        response = chain.invoke({"question": query})
        # 调用模型：
        # question = 原始用户 query
        # 模型返回扩展后的多个查询（通常用分隔符连接）

        response = response.content
        # 提取模型输出文本内容

        queries = response.strip().split(query_expansion_template.separator)
        # 解析模型输出：
        # 1) 去除整体首尾空白
        # 2) 按模板定义的 separator 分割为多个查询字符串

        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip(" \\n"))
        ]
        # 清洗结果：
        # 1) 去除每个查询首尾空格和换行
        # 2) 过滤掉空字符串（使用海象运算符在表达式中赋值）
        # 保证返回的查询列表干净且有效

        return stripped_queries  # 返回扩展后的查询列表（长度通常接近 to_expand_to_n）
