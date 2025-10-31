from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import ArticleChunkModel, PostChunkModel, RepositoryChunkModel
from models.embedded_chunk import (
    ArticleEmbeddedChunkModel,
    PostEmbeddedChunkModel,
    RepositoryEmbeddedChunkModel,
)
from utils.embeddings import embedd_text


class EmbeddingDataHandler(ABC):
    """
    抽象基类：定义“向量化处理器（Embedding Handler）”的统一接口。

    设计目的：
    - 在数据清洗 + 分块之后，需要将文本内容转化为数值向量（embedding）。
    - 不同数据类型（帖子、文章、仓库）虽然字段略有不同，但都共享相同的逻辑：
        “读取 chunk 文本 → 生成 embedding → 输出对应的 EmbeddedChunkModel”。
    - 因此用抽象类统一接口，便于后续扩展或替换 embedding 模型（如 OpenAI, HuggingFace, 自定义模型）。
    """

    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        """
        抽象方法：接收一个 ChunkModel，返回对应的 EmbeddedChunkModel。

        输入：DataModel（通常为 *ChunkModel）
        输出：DataModel（通常为 *EmbeddedChunkModel）
        """
        pass


class PostEmbeddingHandler(EmbeddingDataHandler):
    """
    处理“社交媒体帖子（Post）”类型数据的向量化。

    输入：PostChunkModel（包含 chunk_id 与 chunk_content）
    输出：PostEmbeddedChunkModel（新增 embedded_content 向量字段）
    """

    def embedd(self, data_model: PostChunkModel) -> PostEmbeddedChunkModel:
        return PostEmbeddedChunkModel(
            entry_id=data_model.entry_id,                           # 原始记录 ID
            platform=data_model.platform,                           # 平台名称
            chunk_id=data_model.chunk_id,                           # 分块 ID（来自 chunk 阶段）
            chunk_content=data_model.chunk_content,                 # 分块文本
            embedded_content=embedd_text(data_model.chunk_content), # ✅ 生成文本向量
            author_id=data_model.author_id,                         # 作者 ID
            type=data_model.type,                                   # 数据类型（如 'vector_posts'）
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    """
    处理“文章（Article）”类型数据的向量化。
    """

    def embedd(self, data_model: ArticleChunkModel) -> ArticleEmbeddedChunkModel:
        return ArticleEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            link=data_model.link,                                   # 原始文章链接
            chunk_content=data_model.chunk_content,                 # 文本内容
            chunk_id=data_model.chunk_id,
            embedded_content=embedd_text(data_model.chunk_content), # 生成 embedding 向量
            author_id=data_model.author_id,
            type=data_model.type,
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    """
    处理“代码仓库（Repository）”类型数据的向量化。
    通常仓库的 chunk 内容来自 README、代码注释、文档等。
    """

    def embedd(self, data_model: RepositoryChunkModel) -> RepositoryEmbeddedChunkModel:
        return RepositoryEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            name=data_model.name,
            link=data_model.link,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content), # 将 chunk 文本转化为 embedding 向量
            owner_id=data_model.owner_id,
            type=data_model.type,
        )
