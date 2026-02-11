from config import settings  # 项目配置：读取 OpenAI 模型 ID 和 API Key
from langchain_openai import ChatOpenAI  # LangChain 封装的 OpenAI Chat 模型接口

from core.rag.prompt_templates import RerankingTemplate  # 重排用的 Prompt 模板封装


class Reranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()  # 实例化重排 Prompt 模板对象（内部定义了 prompt 结构与分隔符）
        prompt = reranking_template.create_template(keep_top_k=keep_top_k)  
        # 根据 keep_top_k 动态生成 prompt 模板（通常会在指令中要求模型只返回前 K 个最相关段落）

        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY
        )
        # 初始化 LLM：使用配置文件中的模型 ID 和 API Key（如 gpt-4 / gpt-4o 等）

        chain = prompt | model  
        # 使用 LangChain 的 LCEL 表达式语法：prompt 输出直接作为 model 输入（构建执行链）

        stripped_passages = [
            stripped_item for item in passages if (stripped_item := item.strip())
        ]
        # 预处理 passages：
        # 1) 去掉每个 passage 首尾空白
        # 2) 过滤掉空字符串（利用海象运算符 := 在表达式中赋值）
        # 这样可以避免空段落干扰 LLM 重排结果

        passages = reranking_template.separator.join(stripped_passages)
        # 用模板中定义的 separator（分隔符）将多个段落拼接成一个字符串
        # 模型会根据该分隔符识别不同 passage

        response = chain.invoke({"question": query, "passages": passages})
        # 调用 LLM：
        # question = 原始查询
        # passages = 拼接后的候选段落文本
        # LLM 负责根据相关性重新排序并返回结果

        response = response.content  
        # 提取模型返回的文本内容（LangChain Response 对象中通常包含 metadata）

        reranked_passages = response.strip().split(reranking_template.separator)
        # 对模型输出进行解析：
        # 1) 去除整体首尾空白
        # 2) 按 separator 拆分为多个 passage
        # 假设模型严格按分隔符输出

        stripped_passages = [
            stripped_item
            for item in reranked_passages
            if (stripped_item := item.strip())
        ]
        # 再次清洗：
        # 去除拆分后每个 passage 的空白
        # 过滤空字符串（保证返回结果干净）

        return stripped_passages  # 返回最终重排后的 top-k passages（按相关性排序）
