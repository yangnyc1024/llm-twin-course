import concurrent.futures  # 并发工具：用于线程池并行执行多个检索任务（I/O 密集型很合适）

import opik  # 可观测性/追踪：用于对关键函数进行 trace（如检索、重排）
from config import settings  # 配置：读取 embedding 模型 ID 等运行参数
from qdrant_client import models  # Qdrant 数据模型：Filter / FieldCondition / MatchValue 等
from sentence_transformers.SentenceTransformer import SentenceTransformer  # 向量化：把 query 编码成 embedding

import core.logger_utils as logger_utils  # 日志工具：获取结构化 logger
from core import lib  # 通用工具库：这里用到 flatten 把嵌套 list 拉平
from core.db.qdrant import QdrantDatabaseConnector  # Qdrant 连接器：封装 search 等操作
from core.rag.query_expanison import QueryExpansion  # Query Expansion：把原始 query 扩展为多个检索 query（注意模块名 query_expanison 可能是拼写沿用）
from core.rag.reranking import Reranker  # Reranker：对召回结果做重排，提升相关性
from core.rag.self_query import SelfQuery  # SelfQuery：从 query 中抽取结构化元信息（这里用于提取 author_id）

logger = logger_utils.get_logger(__name__)  # 初始化 logger：按模块名区分日志来源


class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and Multitenancy search.
    """

    def __init__(self, query: str) -> None:
        self._client = QdrantDatabaseConnector()  # Qdrant 客户端封装：用于向量检索
        self.query = query  # 原始用户 query：后续用于扩展、抽取元信息、重排
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)  # embedding 模型：把文本编码成向量
        self._query_expander = QueryExpansion()  # query 扩展器：生成多个改写/扩展 query，提高召回
        self._metadata_extractor = SelfQuery()  # 元信息抽取：从 query 中提取 author_id 等过滤条件
        self._reranker = Reranker()  # 重排器：把召回的文档按相关性再排序

    def _search_single_query(self, generated_query: str, author_id: str, k: int):
        assert k > 3, "k should be greater than 3"  # 约束：下面会用 k//3 分配到 3 个 collection，k 太小会导致每类几乎没结果

        query_vector = self._embedder.encode(generated_query).tolist()  # 将单个扩展 query 编码为向量（并转为 python list 供 Qdrant 使用）

        vectors = [
            self._client.search(
                collection_name="vector_posts",  # 检索 collection：帖子向量库
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="author_id",  # 过滤字段：作者 ID（posts/articles 用 author_id）
                            match=models.MatchValue(
                                value=author_id,  # 过滤值：只看该作者的数据
                            ),
                        )
                    ]
                    if author_id  # 如果抽取到了 author_id 才加过滤；否则不加过滤（= 全局检索）
                    else None
                ),
                query_vector=query_vector,  # 检索向量：用于相似度搜索
                limit=k // 3,  # 每个 collection 分配 k/3 个召回名额（3 个库各取一部分）
            ),
            self._client.search(
                collection_name="vector_articles",  # 检索 collection：文章向量库
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="author_id",  # 过滤字段：作者 ID
                            match=models.MatchValue(
                                value=author_id,
                            ),
                        )
                    ]
                    if author_id
                    else None
                ),
                query_vector=query_vector,
                limit=k // 3,
            ),
            self._client.search(
                collection_name="vector_repositories",  # 检索 collection：代码仓库向量库
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="owner_id",  # 注意：repositories 用的是 owner_id（与 posts/articles 字段不同）
                            match=models.MatchValue(
                                value=author_id,
                            ),
                        )
                    ]
                    if author_id
                    else None
                ),
                query_vector=query_vector,
                limit=k // 3,
            ),
        ]

        return lib.flatten(vectors)  # 将三个 collection 的返回结果合并为一个 list（拍平嵌套结构）

    @opik.track(name="retriever.retrieve_top_k")  # trace：记录检索阶段的输入/输出、耗时等（便于观测与调试）
    def retrieve_top_k(self, k: int, to_expand_to_n_queries: int) -> list:
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries  # 基于原 query 生成 N 个扩展 query
        )
        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(generated_queries),  # 结构化字段：扩展 query 的数量
        )

        author_id = self._metadata_extractor.generate_response(self.query)  # 从原 query 抽取 author_id（用于多租户/按作者过滤）
        if author_id:
            logger.info(
                "Successfully extracted the author_id from the query.",
                author_id=author_id,  # 结构化字段：记录抽取到的 author_id
            )
        else:
            logger.warning("Did not found any author data in the user's prompt.")  # 未抽取到 author 相关信息：降级为全局检索

        with concurrent.futures.ThreadPoolExecutor() as executor:  # 线程池：并发执行多个扩展 query 的检索（通常是 I/O bound）
            search_tasks = [
                executor.submit(self._search_single_query, query, author_id, k)  # 提交任务：每个扩展 query 跑一次 _search_single_query
                for query in generated_queries
            ]

            hits = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)  # 按完成顺序收集结果（避免慢任务阻塞全部）
            ]
            hits = lib.flatten(hits)  # 多个 query 的结果拍平到一个 list（召回集合）

        logger.info("All documents retrieved successfully.", num_documents=len(hits))  # 记录总召回数（注意可能包含重复/跨 query 重叠）

        return hits  # 返回召回的 hits（通常为 Qdrant 的 ScoredPoint 列表或类似结构）

    @opik.track(name="retriever.rerank")  # trace：记录重排阶段（输入 passages 数量、输出 top-k 等）
    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        content_list = [hit.payload["content"] for hit in hits]  # 从 hits 中抽取文本内容（reranker 输入通常是 passages 文本）
        rerank_hits = self._reranker.generate_response(
            query=self.query, passages=content_list, keep_top_k=keep_top_k  # 使用原始 query 对 passages 重排并截断到 top-k
        )

        logger.info("Documents reranked successfully.", num_documents=len(rerank_hits))  # 记录重排输出数量（理论上=keep_top_k 或更少）

        return rerank_hits  # 返回重排后的结果（这里类型标注为 list[str]，推测返回的是 top passages 文本或其标识）

    def set_query(self, query: str):
        self.query = query  # 更新当前 query：便于复用同一个 retriever 实例处理多次查询
