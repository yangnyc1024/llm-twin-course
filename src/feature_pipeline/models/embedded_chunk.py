from typing import Tuple
import numpy as np

from models.base import VectorDBDataModel


class PostEmbeddedChunkModel(VectorDBDataModel):
    """
    ✅ 已向量化的帖子分片数据模型
    用于保存社交媒体帖子（如 LinkedIn、Twitter）的文本片段经过 embedding 后的结构化数据。

    继承自 VectorDBDataModel，必须实现 `to_payload()` 方法，
    以便写入 Qdrant 等向量数据库。
    """

    entry_id: str                # 原始帖子 ID
    platform: str                # 来源平台（如 "linkedin"）
    chunk_id: str                # 当前分片的唯一标识符
    chunk_content: str           # 分片的原始文本内容
    embedded_content: np.ndarray # 向量化后的内容（embedding 向量）
    author_id: str               # 作者 ID
    type: str                    # 数据类型（如 "posts"）

    class Config:
        # 允许 Pydantic 接受非标准类型（如 numpy.ndarray）
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        """
        将对象转换为 (id, vector, payload) 的三元组格式，
        用于写入向量数据库（如 Qdrant）。
        - id: chunk_id
        - vector: 嵌入向量 (embedded_content)
        - payload: 元数据字典 (metadata)
        """
        data = {
            "id": self.entry_id,            # 原始数据 ID
            "platform": self.platform,      # 数据来源
            "content": self.chunk_content,  # 分片文本内容
            "owner_id": self.author_id,     # 作者 ID
            "type": self.type,              # 数据类型
        }

        return self.chunk_id, self.embedded_content, data



class ArticleEmbeddedChunkModel(VectorDBDataModel):
    """
    ✅ 已向量化的文章分片数据模型
    用于保存网站或博客文章分片（chunk）及其对应的 embedding 向量。
    """

    entry_id: str                # 原始文章 ID
    platform: str                # 来源平台（如 "medium"）
    link: str                    # 文章 URL
    chunk_id: str                # 当前分片唯一 ID
    chunk_content: str           # 分片文本内容
    embedded_content: np.ndarray # embedding 向量
    author_id: str               # 作者 ID
    type: str                    # 数据类型（如 "articles"）

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        """
        将文章分片转换为 (id, vector, payload) 格式，
        方便写入向量数据库。
        """
        data = {
            "id": self.entry_id,
            "platform": self.platform,
            "content": self.chunk_content,
            "link": self.link,
            "author_id": self.author_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data



class RepositoryEmbeddedChunkModel(VectorDBDataModel):
    """
    ✅ 已向量化的仓库文档分片数据模型
    用于保存 GitHub 仓库中的文档（如 README、代码注释）在向量化后的结构。
    """

    entry_id: str                # 仓库 ID
    name: str                    # 仓库名称
    link: str                    # 仓库 URL
    chunk_id: str                # 分片 ID
    chunk_content: str           # 分片文本内容
    embedded_content: np.ndarray # 嵌入向量
    owner_id: str                # 仓库所有者 ID
    type: str                    # 数据类型（如 "repositories"）

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        """
        将仓库分片转换为 (id, vector, payload) 格式。
        """
        data = {
            "id": self.entry_id,
            "name": self.name,
            "content": self.chunk_content,
            "link": self.link,
            "owner_id": self.owner_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data





# Crawler（爬虫） → Cleaner（文本清洗）
#       ↓
# Chunker（文本切分） → Embedder（生成向量）
#       ↓
# EmbeddedChunkModel（本文件）
#       ↓
# Qdrant Sink（写入向量数据库）