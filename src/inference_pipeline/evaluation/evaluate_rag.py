import argparse

from core.config import settings
from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts
from llm_twin import LLMTwin
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    ContextPrecision,
    ContextRecall,
    Hallucination,
)

# ✅ 将配置中的远程地址改写为本地回环地址，以适配本地调试场景（如未在容器中运行时）
#    注意：仅用于本地开发调试。上线或 Docker 内运行时请移除此调用（见下方 logger 提示）。
settings.patch_localhost()

logger = get_logger(__name__)
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)
# ↑ 日志提示：当前已对 settings 做了本地化补丁，部署/容器环境请移除。


def evaluation_task(x: dict) -> dict:
    # 评估任务函数：给定一条样本（instruction/content），走一遍真实推理管线并返回评估所需字段
    # 输入字典示例：
    #   {
    #       "instruction": "用户指令/问题",
    #       "content": "参考答案（期望输出）"
    #   }
    # 输出字典需包含：
    #   - input:       模型的输入（问题/指令）
    #   - output:      模型生成的答案
    #   - context:     RAG 检索到的上下文（用于 Hallucination/Context 指标评估）
    #   - expected_output/reference: 数据集中提供的标准答案/参考文本
    inference_pipeline = LLMTwin(mock=False)  # 初始化 LLM Twin 推理管线（关闭 mock，走真实流程）
    result = inference_pipeline.generate(
        query=x["instruction"],  # 将样本中的指令作为查询
        enable_rag=True,         # 启用 RAG：检索外部知识作为上下文辅助作答
    )
    answer = result["answer"]    # 模型生成的答案
    context = result["context"]  # 模型使用到的上下文（检索结果/证据片段）

    return {
        "input": x["instruction"],  # 评估输入（问题）
        "output": answer,           # 模型输出（答案）
        "context": context,         # 上下文（用于幻觉/召回/精度评估）
        "expected_output": x["content"],  # 期望输出（金标准答案）
        "reference": x["content"],        # 同 expected_output，部分评估器使用 reference 字段
    }


def main() -> None:
    # 主流程：解析命令行参数 → 准备数据集 → 配置评估指标 → 执行 evaluate
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")  # 命令行解析器
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="LLMTwinMonitoringDataset",
        help="Name of the dataset to evaluate",  # 将要评估的数据集名称（用于日志/标识）
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name  # 读取参数中的数据集名称（此处主要用于日志展示）

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")  # 打印评估目标数据集名

    # 从已产出的工件（artifacts）构建评估数据集
    # 这里会将三类指令数据（文章/动态/代码仓库）汇总为一个 Opik 数据集
    dataset = create_dataset_from_artifacts(
        dataset_name="LLMTwinArtifactTestDataset",  # 实际创建/使用的 Opik 数据集名
        artifact_names=[
            "articles-instruct-dataset",      # 文章类指令-答案对
            "posts-instruct-dataset",         # 帖子类指令-答案对
            "repositories-instruct-dataset",  # 代码仓库类指令-答案对
        ],
    )
    if dataset is None:
        # 若工件不存在或装载失败，则无法继续评估
        logger.error("Dataset can't be created. Exiting.")
        exit(1)

    # 实验配置：指定生成模型与嵌入模型（供检索/相似度等使用）
    experiment_config = {
        "model_id": settings.MODEL_ID,                    # 文本生成模型 ID
        "embedding_model_id": settings.EMBEDDING_MODEL_ID # 向量化/检索所用的嵌入模型 ID
    }

    # 评估指标集合：
    # - Hallucination：检测答案是否脱离提供的 context（编造/不一致）
    # - ContextRecall：检索召回率（参考答案中的关键信息，被 context 覆盖的比例）
    # - ContextPrecision：检索精度（context 中的内容有多少与答案/问题相关）
    scoring_metrics = [
        Hallucination(),
        ContextRecall(),
        ContextPrecision(),
    ]

    # 执行评估：对数据集中每条样本运行 evaluation_task，并按指标打分
    evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=scoring_metrics,
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()  # 程序入口：运行评估流程
