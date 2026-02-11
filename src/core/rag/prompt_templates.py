from abc import ABC, abstractmethod  # ABC：定义抽象基类；abstractmethod：定义抽象方法

from langchain.prompts import PromptTemplate  # LangChain 的 Prompt 模板类
from pydantic import BaseModel  # Pydantic BaseModel：用于数据校验与结构化定义


class BasePromptTemplate(ABC, BaseModel):
    # 抽象 Prompt 模板基类：
    # 1) 继承 ABC：强制子类实现抽象方法
    # 2) 继承 BaseModel：支持字段校验与默认值管理

    @abstractmethod
    def create_template(self, *args) -> PromptTemplate:
        # 抽象方法：所有子类必须实现该方法
        # 作用：返回一个构造好的 LangChain PromptTemplate 实例
        pass


class QueryExpansionTemplate(BasePromptTemplate):
    # Query Expansion 用的 Prompt 模板：
    # 用于让 LLM 生成多个语义相近但表达不同的查询版本

    prompt: str = """You are an AI language model assistant. Your task is to generate {to_expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by '{separator}'.
    Original question: {question}"""
    # Prompt 模板字符串：
    # - {to_expand_to_n}：生成的查询数量
    # - {separator}：不同查询之间的分隔符
    # - {question}：原始用户问题

    @property
    def separator(self) -> str:
        # 定义扩展查询之间的分隔符
        # 模型必须严格按照该分隔符输出，方便后续 split 解析
        return "#next-question#"

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        # 构建 LangChain PromptTemplate
        return PromptTemplate(
            template=self.prompt,  # 使用类中定义的 prompt 字符串
            input_variables=["question"],  # 运行时必须提供的变量
            partial_variables={
                "separator": self.separator,  # 固定注入分隔符
                "to_expand_to_n": to_expand_to_n,  # 固定注入扩展数量
            },
        )


class SelfQueryTemplate(BasePromptTemplate):
    # SelfQuery 用的 Prompt 模板：
    # 目标是从自然语言问题中抽取 user name 或 user id

    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user name or user id. 
    Your response should consists of only the extracted user name (e.g., John Doe) or id (e.g. 1345256), nothing else.
    If the user question does not contain any user name or id, you should return the following token: none.
    
    For example:
    QUESTION 1:
    My name is Paul Iusztin and I want a post about...
    RESPONSE 1:
    Paul Iusztin
    
    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    none
    
    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256
    
    User question: {question}"""
    # Prompt 模板逻辑：
    # - 明确只允许输出姓名或 ID
    # - 若不存在则返回 "none"
    # - 提供 few-shot 示例帮助模型对齐格式

    def create_template(self) -> PromptTemplate:
        # 构建 PromptTemplate，仅需 question 变量
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class RerankingTemplate(BasePromptTemplate):
    # Reranking 用的 Prompt 模板：
    # 目标是根据 query 对 passages 进行相关性重排

    prompt: str = """You are an AI language model assistant. Your task is to rerank passages related to a query
    based on their relevance. 
    The most relevant passages should be put at the beginning. 
    You should only pick at max {keep_top_k} passages.
    The provided and reranked documents are separated by '{separator}'.
    
    The following are passages related to this query: {question}.
    
    Passages: 
    {passages}
    """
    # 模板变量说明：
    # - {keep_top_k}：最多保留的段落数量
    # - {separator}：段落之间的分隔符
    # - {question}：原始查询
    # - {passages}：候选段落列表（拼接后的字符串）

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        # 构建 PromptTemplate，用于重排阶段
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],  # 运行时必须传入 query 和候选段落
            partial_variables={"keep_top_k": keep_top_k, "separator": self.separator},
            # 固定注入：
            # - keep_top_k：限制返回数量
            # - separator：用于解析模型输出
        )

    @property
    def separator(self) -> str:
        # 定义文档之间的分隔符
        # 使用换行 + 标记，增强可读性并减少误切分风险
        return "\n#next-document#\n"
