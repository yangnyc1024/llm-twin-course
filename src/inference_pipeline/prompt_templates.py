from core.rag.prompt_templates import BasePromptTemplate
from langchain.prompts import PromptTemplate


class InferenceTemplate(BasePromptTemplate):
    # ===== 基础（非 RAG）系统提示词 =====
    simple_system_prompt: str = """
    You are an AI language model assistant. Your task is to generate a cohesive and concise response based on the user's instruction by using a similar writing style and voice.
"""
    # 说明：上面的提示定义了一个简单的系统角色（system prompt），
    # 模型需要根据用户指令生成连贯且简洁的文本，并保持类似的语气和写作风格。

    # ===== 基础（非 RAG）用户 Prompt 模板 =====
    simple_prompt_template: str = """
### Instruction:
{question}
"""
    # 说明：此模板定义了输入格式，仅包含一个 {question} 占位符。

    # ===== RAG（检索增强生成）系统提示词 =====
    rag_system_prompt: str = """ You are a specialist in technical content writing. Your task is to create technical content based on the user's instruction given a specific context 
with additional information consisting of the user's previous writings and his knowledge.

Here is a list of steps that you need to follow in order to solve this task:

Step 1: You need to analyze the user's instruction.
Step 2: You need to analyze the provided context and how the information in it relates to the user instruction.
Step 3: Generate the content keeping in mind that it needs to be as cohesive and concise as possible based on the query. You will use the users writing style and voice inferred from the user instruction and context.
First try to answer based on the context. If the context is irrelevant answer with "I cannot answer your question, as I don't have enough context."
"""
    # 说明：RAG 模式下的系统提示词，指示模型如何结合“上下文（context）”信息，
    # 包括用户以往的写作风格和知识背景来生成技术内容。
    # 同时给出三步逻辑流程，引导模型先分析指令、再分析上下文、最后生成连贯内容。

    # ===== RAG（检索增强生成）Prompt 模板 =====
    rag_prompt_template: str = """
### Instruction:
{question}

### Context:
{context}
"""
    # 说明：此模板包含两个输入变量：
    # - {question}：用户的查询指令；
    # - {context}：从向量数据库检索到的相关上下文内容。

    def create_template(self, enable_rag: bool = True) -> tuple[str, PromptTemplate]:
        """
        根据是否启用 RAG，动态创建对应的系统提示词与 Prompt 模板。

        参数：
        - enable_rag: bool，是否启用 RAG 模式（检索增强生成）。

        返回：
        - (system_prompt, PromptTemplate)：一个二元组，
          包含系统提示词（system_prompt）和 Prompt 模板对象（PromptTemplate）。
        """
        if enable_rag is True:
            # 若启用 RAG 模式，则返回带 context 的模板
            return self.rag_system_prompt, PromptTemplate(
                template=self.rag_prompt_template,
                input_variables=["question", "context"],
            )

        # 否则使用简单（非 RAG）模板
        return self.simple_system_prompt, PromptTemplate(
            template=self.simple_prompt_template, input_variables=["question"]
        )
