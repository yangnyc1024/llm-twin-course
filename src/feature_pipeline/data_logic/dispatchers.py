from core import get_logger
from models.base import DataModel
from models.raw import ArticleRawModel, PostsRawModel, RepositoryRawModel

# 导入三个阶段（clean → chunk → embed）的处理器类
from data_logic.chunking_data_handlers import (
    ArticleChunkingHandler,
    ChunkingDataHandler,
    PostChunkingHandler,
    RepositoryChunkingHandler,
)
from data_logic.cleaning_data_handlers import (
    ArticleCleaningHandler,
    CleaningDataHandler,
    PostCleaningHandler,
    RepositoryCleaningHandler,
)
from data_logic.embedding_data_handlers import (
    ArticleEmbeddingHandler,
    EmbeddingDataHandler,
    PostEmbeddingHandler,
    RepositoryEmbeddingHandler,
)

logger = get_logger(__name__)


# ============================================================
# 一、RawDispatcher：负责从消息队列（MQ）中接收原始数据
# ============================================================

class RawDispatcher:
    """
    根据 MQ 消息内容（JSON dict）解析出正确的数据模型（RawModel）。

    功能：
    - 负责接收来自 RabbitMQ 或其他消息队列的原始数据；
    - 根据 message["type"] 字段（posts / articles / repositories）自动实例化对应的 RawModel；
    - 起到「统一入口 + 类型识别」的作用。
    """

    @staticmethod
    def handle_mq_message(message: dict) -> DataModel:
        data_type = message.get("type")

        logger.info("Received message.", data_type=data_type)

        # 根据类型动态创建对应的原始数据模型
        if data_type == "posts":
            return PostsRawModel(**message)
        elif data_type == "articles":
            return ArticleRawModel(**message)
        elif data_type == "repositories":
            return RepositoryRawModel(**message)
        else:
            raise ValueError("Unsupported data type")


# ============================================================
# 二、Cleaning 阶段：工厂 + 分发器
# ============================================================

class CleaningHandlerFactory:
    """
    清洗处理器（Cleaning Handler）的工厂类。
    根据数据类型创建相应的清洗类实例。
    """

    @staticmethod
    def create_handler(data_type) -> CleaningDataHandler:
        if data_type == "posts":
            return PostCleaningHandler()
        elif data_type == "articles":
            return ArticleCleaningHandler()
        elif data_type == "repositories":
            return RepositoryCleaningHandler()
        else:
            raise ValueError("Unsupported data type")


class CleaningDispatcher:
    """
    调度清洗逻辑的调度器类。
    用于调用正确的清洗 Handler 执行数据清洗。

    输入：RawModel
    输出：CleanedModel
    """

    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch_cleaner(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        clean_model = handler.clean(data_model)

        logger.info(
            "Data cleaned successfully.",
            data_type=data_type,
            cleaned_content_len=len(clean_model.cleaned_content),
        )

        return clean_model


# ============================================================
# 三、Chunking 阶段：工厂 + 分发器
# ============================================================

class ChunkingHandlerFactory:
    """
    分块处理器（Chunking Handler）的工厂类。
    根据数据类型返回对应的 ChunkingHandler。
    """

    @staticmethod
    def create_handler(data_type) -> ChunkingDataHandler:
        if data_type == "posts":
            return PostChunkingHandler()
        elif data_type == "articles":
            return ArticleChunkingHandler()
        elif data_type == "repositories":
            return RepositoryChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    """
    调度分块逻辑的调度器类。
    用于执行数据切分（chunking）。

    输入：CleanedModel
    输出：list[ChunkModel]
    """

    cleaning_factory = ChunkingHandlerFactory

    @classmethod
    def dispatch_chunker(cls, data_model: DataModel) -> list[DataModel]:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Cleaned content chunked successfully.",
            num=len(chunk_models),
            data_type=data_type,
        )

        return chunk_models


# ============================================================
# 四、Embedding 阶段：工厂 + 分发器
# ============================================================

class EmbeddingHandlerFactory:
    """
    向量化处理器（Embedding Handler）的工厂类。
    根据数据类型返回对应的 EmbeddingHandler。
    """

    @staticmethod
    def create_handler(data_type) -> EmbeddingDataHandler:
        if data_type == "posts":
            return PostEmbeddingHandler()
        elif data_type == "articles":
            return ArticleEmbeddingHandler()
        elif data_type == "repositories":
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    """
    调度 embedding 逻辑的调度器类。
    用于执行文本向量化。

    输入：ChunkModel
    输出：EmbeddedChunkModel
    """

    cleaning_factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch_embedder(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        embedded_chunk_model = handler.embedd(data_model)

        logger.info(
            "Chunk embedded successfully.",
            data_type=data_type,
            embedding_len=len(embedded_chunk_model.embedded_content),
        )

        return embedded_chunk_model