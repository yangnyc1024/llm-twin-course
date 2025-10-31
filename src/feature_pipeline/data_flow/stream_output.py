from bytewax.outputs import DynamicSink, StatelessSinkPartition
from core import get_logger
from core.db.qdrant import QdrantDatabaseConnector
from models.base import VectorDBDataModel
from qdrant_client.models import Batch

logger = get_logger(__name__)


class QdrantOutput(DynamicSink):
    """
    Bytewax 的输出端（Sink）定义：连接 Qdrant 向量数据库。
    之所以继承 DynamicSink，是因为我们需要“根据不同数据类型/用途”
    动态构建不同的 sink 分区（比如：写入纯清洗数据的集合 vs 写入向量集合）。
    """

    def __init__(self, connection: QdrantDatabaseConnector, sink_type: str):
        # 注入 Qdrant 连接器实例 & 该 Sink 的类型（'clean' 或 'vector'）
        self._connection = connection
        self._sink_type = sink_type

        # 这里定义了所有会用到的 Qdrant 集合，以及它们是否为“向量集合”
        collections = {
            "cleaned_posts": False,
            "cleaned_articles": False,
            "cleaned_repositories": False,
            "vector_posts": True,
            "vector_articles": True,
            "vector_repositories": True,
        }

        # 确保上述集合已经存在；若不存在则创建
        for collection_name, is_vector in collections.items():
            try:
                self._connection.get_collection(collection_name=collection_name)
            except Exception:
                logger.warning(
                    "Couldn't access the collection. Creating a new one...",
                    collection_name=collection_name,
                )
                # 根据集合类型分别创建：向量集合 or 非向量集合
                if is_vector:
                    self._connection.create_vector_collection(
                        collection_name=collection_name
                    )
                else:
                    self._connection.create_non_vector_collection(
                        collection_name=collection_name
                    )

    def build(self, worker_index: int, worker_count: int) -> StatelessSinkPartition:
        """
        Bytewax 会为每个执行 worker 调用 build() 来构造实际的 Sink 分区实例。
        此处根据 sink_type 返回不同的写入策略：
        - 'clean'  → 写入 cleaned_* 集合（只写 payload/元数据）
        - 'vector' → 写入 vector_* 集合（写向量 + 元数据）
        """
        if self._sink_type == "clean":
            return QdrantCleanedDataSink(connection=self._connection)
        elif self._sink_type == "vector":
            return QdrantVectorDataSink(connection=self._connection)
        else:
            raise ValueError(f"Unsupported sink type: {self._sink_type}")


class QdrantCleanedDataSink(StatelessSinkPartition):
    """
    写入“清洗后的数据”的 Sink 分区实现：
    - 不包含向量，仅写入 payload（结构化字段，如 source、url、text、type 等）
    - 目标集合为 cleaned_posts / cleaned_articles / cleaned_repositories
    """

    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        """
        Bytewax 会在流水线中把一个 batch 的数据交给 write_batch。
        这里将模型对象转成 Qdrant 批量写入所需的 (ids, vectors, payloads) 结构。
        对于 cleaned_* 集合，vectors 传空字典 {}。
        """
        # VectorDBDataModel.to_payload() 在“clean”场景应返回 (id, payload)
        payloads = [item.to_payload() for item in items]
        ids, data = zip(*payloads)

        # 根据第一条数据的 type 字段，动态选集合
        collection_name = get_clean_collection(data_type=data[0]["type"])

        # 通过 Qdrant 的 Batch API 写入：此处 vectors 为空
        self._client.write_data(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors={}, payloads=data),
        )

        logger.info(
            "Successfully inserted requested cleaned point(s)",
            collection_name=collection_name,
            num=len(ids),
        )


class QdrantVectorDataSink(StatelessSinkPartition):
    """
    写入“向量数据”的 Sink 分区实现：
    - 同时写入向量和 payload
    - 目标集合为 vector_posts / vector_articles / vector_repositories
    """

    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        """
        与 clean 数据不同，这里需要传 (id, vector, metadata) 三元组，
        其中 metadata 作为 payload 存入，vector 存放在 vectors 字段中。
        """
        # VectorDBDataModel.to_payload() 在“vector”场景应返回 (id, vector, metadata)
        payloads = [item.to_payload() for item in items]
        ids, vectors, meta_data = zip(*payloads)

        # 依据数据类型选择向量集合
        collection_name = get_vector_collection(data_type=meta_data[0]["type"])

        # 批量写入向量与元数据
        self._client.write_data(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=meta_data),
        )

        logger.info(
            "Successfully inserted requested vector point(s)",
            collection_name=collection_name,
            num=len(ids),
        )


def get_clean_collection(data_type: str) -> str:
    """
    将自定义的数据类型（type 字段）映射到具体 cleaned_* 集合名。
    """
    if data_type == "posts":
        return "cleaned_posts"
    elif data_type == "articles":
        return "cleaned_articles"
    elif data_type == "repositories":
        return "cleaned_repositories"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_vector_collection(data_type: str) -> str:
    """
    将自定义的数据类型（type 字段）映射到具体 vector_* 集合名。
    """
    if data_type == "posts":
        return "vector_posts"
    elif data_type == "articles":
        return "vector_articles"
    elif data_type == "repositories":
        return "vector_repositories"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")



# ┌──────────────────────────────────────────────┐
# │                Feature Pipeline              │
# │                                              │
# │  ┌──────────────┐       ┌────────────┐       │
# │  │ stream_input │──────▶│ stream_logic│──────▶│
# │  │ (RabbitMQ→)  │       │ (清洗+嵌入)│       │
# │  └──────────────┘       └────────────┘       │
# │          │                      │            │
# │          │                      │            │
# │          ▼                      ▼            │
# │  QdrantCleanedDataSink   QdrantVectorDataSink│
# │          │                      │            │
# │          ▼                      ▼            │
# │  cleaned_posts/...        vector_posts/...   │
# └──────────────────────────────────────────────┘


# Qdrant向量， key, vectors, meta
# [
#   {
#     "id": 1,
#     "vector": [0.1, 0.2, 0.3, 0.4],
#     "payload": {"type": "post", "content": "Cybersecurity tips for small businesses"}
#   },
#   {
#     "id": 2,
#     "vector": [0.9, 0.8, 0.1, 0.3],
#     "payload": {"type": "article", "title": "Understanding cyber risk models"}
#   },
#   {
#     "id": 3,
#     "vector": [0.5, 0.4, 0.6, 0.7],
#     "payload": {"type": "repository", "name": "cyber-risk-analyzer"}
#   }
# ]
