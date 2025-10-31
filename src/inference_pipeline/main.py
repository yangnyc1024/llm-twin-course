import sys
from pathlib import Path

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
# 为了在开发或教学环境中模拟多个 Python 模块（如 core、feature_pipeline）的导入，
# 这里手动将 ./src 目录添加到 Python 的模块搜索路径中。
# 注意：此做法仅适用于本地开发环境，不建议在生产部署中使用。
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
from core.config import settings
from llm_twin import LLMTwin

# 调用自定义方法，修改配置以适配本地环境（如将远程地址改为 localhost）
settings.patch_localhost()

# 初始化日志记录器
logger = logger_utils.get_logger(__name__)
logger.info(
    f"Added the following directory to PYTHONPATH to simulate multiple modules: {ROOT_DIR}"
)
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)
# ⚠️ 提示：在 Docker 或正式部署环境中，应删除 settings.patch_localhost()，
# 因为它仅用于本地测试时修改为 localhost 环境。

if __name__ == "__main__":
    # 初始化 LLMTwin 推理类实例（mock=False 表示实际调用模型端点）
    inference_endpoint = LLMTwin(mock=False)

    # 定义测试查询内容（包含多行字符串）
    query = """
Hello I am Paul Iusztin.
        
Could you draft an article paragraph discussing RAG? 
I'm particularly interested in how to design a RAG system.
        """

    # 调用 LLM 推理流程，启用 RAG 检索增强，并保存样本用于评估
    response = inference_endpoint.generate(
        query=query, enable_rag=True, sample_for_evaluation=True
    )

    # 打印日志信息，展示输入与输出结果
    logger.info("=" * 50)
    logger.info(f"Query: {query}")
    logger.info("=" * 50)
    logger.info(f"Answer: {response['answer']}")
    logger.info("=" * 50)
