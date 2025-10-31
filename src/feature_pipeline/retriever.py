import sys
from pathlib import Path

# ✅ 将项目根目录（src）加入到 Python 模块搜索路径中
# 这样就可以在本地直接运行时，正确导入 core 和 feature_pipeline 等模块。
# 注意：这是为了开发和教学目的使用的临时方法，不建议在生产环境中使用。
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)


# ✅ 导入核心模块
from core import get_logger                       # 项目中统一的日志记录器
from core.config import settings                  # 全局配置管理（使用 pydantic settings）
from core.rag.retriever import VectorRetriever    # 向量检索模块（RAG 的核心组件）

# ✅ 初始化日志对象
logger = get_logger(__name__)

# ✅ 将配置中的主机名替换为 localhost，以便在本地调试（如非 Docker 环境）
settings.patch_localhost()
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)

# ✅ 主程序入口
if __name__ == "__main__":
    # 定义测试查询（用户的问题）
    query = """
Hello I am Paul Iusztin.
        
Could you draft an article paragraph discussing RAG? 
I'm particularly interested in how to design a RAG system.
"""

    # 初始化检索器实例，传入查询语句
    retriever = VectorRetriever(query=query)

    # 步骤 1️⃣：执行向量检索（召回 top-k 相关文档）
    hits = retriever.retrieve_top_k(k=6, to_expand_to_n_queries=5)

    # 步骤 2️⃣：对召回的文档进行重新排序（reranking）
    reranked_hits = retriever.rerank(hits=hits, keep_top_k=5)

    # 步骤 3️⃣：打印最终结果
    logger.info("====== RETRIEVED DOCUMENTS ======")
    for rank, hit in enumerate(reranked_hits):
        logger.info(f"Rank = {rank} : {hit}")
