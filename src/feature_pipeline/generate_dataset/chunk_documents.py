import re


def chunk_documents(documents: list[str], min_length: int = 1000, max_length: int = 2000):
    """
    将一组长文本文档（documents）进行分块（chunking）。

    参数：
        documents (list[str]): 文本列表，每个元素是一整篇清洗后的文本。
        min_length (int): 每个 chunk 的最小长度（字符数），默认 1000。
        max_length (int): 每个 chunk 的最大长度（字符数），默认 2000。

    返回：
        list[str]: 所有分块后的文本列表（每个 chunk 是字符串）。

    逻辑说明：
        - 对每个文档调用 extract_substrings()，根据句子边界分割；
        - 每个 chunk 保持长度介于 [min_length, max_length]；
        - 最终输出一个包含所有文档 chunks 的列表。
    """
    chunked_documents = []

    # 遍历输入的每篇文档
    for document in documents:
        # 调用下方函数 extract_substrings() 执行实际的句子级分块
        chunks = extract_substrings(document, min_length=min_length, max_length=max_length)
        # 将所有分块追加到最终列表
        chunked_documents.extend(chunks)
        
    return chunked_documents


def extract_substrings(
    text: str, min_length: int = 1000, max_length: int = 2000
) -> list[str]:
    """
    根据句子边界将单篇文本分割为若干子串（chunks）。

    参数：
        text (str): 输入的长文本字符串。
        min_length (int): 每个 chunk 的最小长度（字符数）。
        max_length (int): 每个 chunk 的最大长度（字符数）。

    返回：
        list[str]: 由多个分块文本组成的列表。

    工作原理：
        1️⃣ 使用正则表达式按句号 / 问号 / 感叹号后面的空格进行分句；
        2️⃣ 遍历句子，按 max_length 拼接句子到 current_chunk；
        3️⃣ 如果当前块超过 max_length 或已达 min_length，就将其保存；
        4️⃣ 保证每个 chunk 不太短，也不超过上限；
        5️⃣ 适合用于 LLM prompt、embedding 或 RAG chunking。
    """

    # 使用正则表达式按句号、问号、感叹号后分割文本（尽量保留句子完整性）
    # (?<=\.|\?|\!) 表示“在 . ? ! 后面切分”
    # 同时通过 (?<!\w\.\w.) 等负向前瞻避免在缩写、网址、名字中误切（如 U.S.A.）
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []        # 保存最终分块的列表
    current_chunk = ""   # 当前正在构建的分块文本

    for sentence in sentences:
        sentence = sentence.strip()  # 去掉首尾空格
        if not sentence:
            continue  # 跳过空句子

        # 如果加上当前句子后仍未超过 max_length，则继续拼接
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            # 如果当前 chunk 已经够长（超过 min_length），则保存下来
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            # 开启一个新的 chunk
            current_chunk = sentence + " "

    # 最后一个 chunk 可能没满，也要判断是否够长再加入结果
    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
