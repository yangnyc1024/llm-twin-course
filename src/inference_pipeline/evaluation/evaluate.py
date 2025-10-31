import argparse

from config import settings
from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts
from llm_twin import LLMTwin
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, LevenshteinRatio, Moderation

from .style import Style

logger = get_logger(__name__)


def evaluation_task(x: dict) -> dict:
    """
    定义单条样本的评估任务逻辑。
    - 输入：包含 'instruction'（模型输入指令）和 'content'（期望输出内容）。
    - 流程：通过 LLMTwin 模型生成答案。
    - 输出：返回字典格式，包括模型输入、输出及期望答案。
    """
    inference_pipeline = LLMTwin(mock=False)  # 初始化 LLM Twin 推理管线（关闭 mock，启用真实推理）
    result = inference_pipeline.generate(
        query=x["instruction"],  # 将输入指令作为模型的 query
        enable_rag=False,        # 关闭 RAG 检索功能，仅评估模型生成能力
    )
    answer = result["answer"]   # 获取模型输出的答案

    return {
        "input": x["instruction"],       # 模型输入（问题/指令）
        "output": answer,                # 模型输出（答案）
        "expected_output": x["content"], # 期望输出（标准答案/参考文本）
        "reference": x["content"],       # 同 expected_output，供某些指标使用
    }


def main() -> None:
    """
    主函数：加载数据集 → 配置实验参数与指标 → 执行评估。
    """
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")  # 创建命令行参数解析器
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="LLMTwinMonitoringDataset",
        help="Name of the dataset to evaluate",  # 数据集名称参数说明
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name  # 从命令行参数中获取数据集名称

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")  # 输出当前正在评估的数据集名称

    # 通过工件（artifact）生成评估数据集
    # 这里整合多个已存在的 instruct 数据集（文章类与代码仓库类）
    dataset = create_dataset_from_artifacts(
        dataset_name="LLMTwinArtifactTestDataset",  # 在 Opik 中的目标数据集名称
        artifact_names=[
            "articles-instruct-dataset",      # 文章类指令-回答样本
            "repositories-instruct-dataset",  # 代码仓库类指令-回答样本
        ],
    )

    # 若数据集创建失败，则退出程序
    if dataset is None:
        logger.error("Dataset can't be created. Exiting.")
        exit(1)

    # 实验配置参数，用于记录评估模型信息
    experiment_config = {
        "model_id": settings.MODEL_ID,  # 当前评估的模型 ID（从配置中读取）
    }

    # 定义评估指标列表
    # - LevenshteinRatio：编辑距离相似度（衡量输出与参考文本的文本相似度）
    # - Hallucination：幻觉检测指标（判断输出是否编造或与上下文不符）
    # - Moderation：内容合规性检测（检查是否包含违规/不安全内容）
    # - Style：风格一致性指标（自定义模块，评估输出是否符合预期语言风格）
    scoring_metrics = [
        LevenshteinRatio(),
        Hallucination(),
        Moderation(),
        Style(),
    ]

    # 执行评估流程
    evaluate(
        dataset=dataset,                   # 输入评估数据集
        task=evaluation_task,              # 定义任务处理逻辑
        scoring_metrics=scoring_metrics,   # 设定评估指标
        experiment_config=experiment_config,  # 实验参数配置
    )


if __name__ == "__main__":
    main()  # 程序入口：启动评估流程
