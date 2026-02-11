from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch, Distance, VectorParams

import core.logger_utils as logger_utils
from core.config import settings

logger = logger_utils.get_logger(__name__)


class QdrantDatabaseConnector:
    # 中文注释：封装 QdrantClient 的数据库连接类，用于管理向量库相关操作
    _instance: QdrantClient | None = None  # 中文注释：缓存 QdrantClient 实例

    def __init__(self) -> None:
        # 中文注释：在初始化时根据配置决定连接本地 Qdrant 还是 Qdrant Cloud
        if self._instance is None:
            if settings.USE_QDRANT_CLOUD:
                # 中文注释：连接 Qdrant Cloud（通过 URL + API Key）
                self._instance = QdrantClient(
                    url=settings.QDRANT_CLOUD_URL,
                    api_key=settings.QDRANT_APIKEY,
                )
            else:
                # 中文注释：连接本地部署的 Qdrant（通过 host + port）
                self._instance = QdrantClient(
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                )

    def get_collection(self, collection_name: str):
        # 中文注释：获取指定 collection 的元信息（例如向量配置等）
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        # 中文注释：创建一个不包含向量字段的 collection（纯 payload 存储）
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str):
        # 中文注释：创建一个向量 collection，配置向量维度和距离度量方式
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE, distance=Distance.COSINE
            ),  # 中文注释：使用余弦相似度进行向量距离计算
        )

    def write_data(self, collection_name: str, points: Batch):
        # 中文注释：向指定 collection 中写入（或更新）一批向量数据
        try:
            # 中文注释：upsert = update + insert；若存在则更新，不存在则插入
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            # 中文注释：写入失败时记录异常日志，并继续向上抛出异常
            logger.exception("An error occurred while inserting data.")

            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        # 中文注释：执行向量相似度搜索
        # - query_vector: 查询向量
        # - query_filter: 可选的 payload 过滤条件
        # - limit: 返回结果数量
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        # 中文注释：顺序遍历（分页拉取）collection 中的数据
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        # 中文注释：关闭 Qdrant 连接（通常在应用关闭时调用）
        if self._instance:
            self._instance.close()

            logger.info("Connected to database has been closed.")
