import bytewax.operators as op
from bytewax.dataflow import Dataflow
from core.db.qdrant import QdrantDatabaseConnector
from data_flow.stream_input import RabbitMQSource
from data_flow.stream_output import QdrantOutput
from data_logic.dispatchers import (
    ChunkingDispatcher,
    CleaningDispatcher,
    EmbeddingDispatcher,
    RawDispatcher,
)


# ✅ 初始化 Qdrant 数据库连接器
# 用于后续在数据流中写入清洗后或向量化的数据
connection = QdrantDatabaseConnector()


# ✅ 定义 Bytewax 的数据流对象
# 给数据流命名为 "Streaming ingestion pipeline"
flow = Dataflow("Streaming ingestion pipeline")


# ✅ Step 1️⃣: 从 RabbitMQ 读取实时数据
# RabbitMQSource 封装了消息队列的消费逻辑
# 每条消息通常包含从爬虫系统传来的“原始数据” (Raw data)
stream = op.input("input", flow, RabbitMQSource())


# ✅ Step 2️⃣: 解析消息 —— RawDispatcher
# 将 MQ 消息反序列化为对应的 RawModel 对象（如 PostsRawModel / ArticleRawModel）
stream = op.map("raw dispatch", stream, RawDispatcher.handle_mq_message)


# ✅ Step 3️⃣: 数据清洗 —— CleaningDispatcher
# 根据不同数据类型（post/article/repository）调用对应的清洗函数
# 输出的是 CleanedModel（如 PostCleanedModel）
stream = op.map("clean dispatch", stream, CleaningDispatcher.dispatch_cleaner)


# ✅ Step 4️⃣: 写入清洗结果到 Qdrant（非向量集合）
# 将清洗后的数据保存到 Qdrant 的 “cleaned_xxx” 集合中
# sink_type="clean" 表示写入清洗数据集合
op.output(
    "cleaned data insert to qdrant",
    stream,
    QdrantOutput(connection=connection, sink_type="clean"),
)


# ✅ Step 5️⃣: 文本切分 —— ChunkingDispatcher
# 将长文本内容分割为多个较短的 chunk（用于后续 embedding）
# 使用 flat_map 因为每个输入对象可能拆成多个输出 chunk
stream = op.flat_map("chunk dispatch", stream, ChunkingDispatcher.dispatch_chunker)


# ✅ Step 6️⃣: 向量化（Embedding） —— EmbeddingDispatcher
# 对每个文本分片生成 embedding 向量（如 OpenAI Embeddings 或 SentenceTransformers）
# 输出的是 EmbeddedChunkModel（如 PostEmbeddedChunkModel）
stream = op.map(
    "embedded chunk dispatch", stream, EmbeddingDispatcher.dispatch_embedder
)


# ✅ Step 7️⃣: 写入向量数据到 Qdrant
# 将嵌入向量存入 “vector_xxx” 集合中，用于后续语义检索（RAG 模块使用）
op.output(
    "embedded data insert to qdrant",
    stream,
    QdrantOutput(connection=connection, sink_type="vector"),
)



    #  ┌────────────────────────┐
    #  │   RabbitMQ (消息队列)  │  ← 爬虫推送原始数据
    #  └────────────┬───────────┘
    #               │
    #               ▼
    #       [RabbitMQSource]
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ RawDispatcher       │ → 将消息解析成 RawModel
    #    └────────────────────┘
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ CleaningDispatcher  │ → 清洗文本、提取正文
    #    └────────────────────┘
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ QdrantOutput(clean) │ → 写入 “cleaned_xxx” 集合
    #    └────────────────────┘
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ ChunkingDispatcher  │ → 文本切片
    #    └────────────────────┘
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ EmbeddingDispatcher │ → 调用 Embedding 模型生成向量
    #    └────────────────────┘
    #               │
    #               ▼
    #    ┌────────────────────┐
    #    │ QdrantOutput(vector)│ → 写入 “vector_xxx” 集合
    #    └────────────────────┘
