import argparse

import opik
from config import settings
from core.logger_utils import get_logger
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, Hallucination, Moderation

from .style import Style

logger = get_logger(__name__)


def evaluation_task(x: dict) -> dict:
    """
    定义评估任务的输入输出结构。
    输入：包含 query（问题）、context（上下文）、answer（答案）。
    输出：返回字典形式的三要素，用于后续模型评估。
    """
    return {
        "input": x["input"]["query"],          # 模型输入的 query（问题）
        "context": x["expected_output"]["context"],  # 评估时使用的上下文内容
        "output": x["expected_output"]["answer"],    # 模型生成或期望的答案
    }


def main() -> None:
    """
    主函数：用于加载 Opik 监控数据集并执行评估任务。
    """
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")  # 创建命令行解析器
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="LLMTwinMonitoringDataset",
        help="Name of the dataset to evaluate",  # 数据集名称参数
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name  # 从命令行参数获取数据集名称

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")  # 打印当前正在评估的数据集名称

    client = opik.Opik()  # 初始化 Opik 客户端实例（用于访问数据集）
    try:
        dataset = client.get_dataset(dataset_name)  # 从 Opik 获取指定名称的数据集
    except Exception:
        logger.error(f"Monitoring dataset '{dataset_name}' not found in Opik. Exiting.")  # 如果不存在，打印错误并退出
        exit(1)

    # 设定实验配置（主要包括模型 ID）
    experiment_config = {
        "model_id": settings.MODEL_ID,
    }

    # 定义要使用的评分指标
    scoring_metrics = [
        Hallucination(),   # 幻觉检测指标
        Moderation(),      # 内容合规性指标
        AnswerRelevance(), # 答案相关性指标
        Style(),           # 自定义的风格一致性指标（本地模块）
    ]

    # 调用 Opik 的 evaluate 方法执行评估
    evaluate(
        dataset=dataset,                   # 输入的数据集
        task=evaluation_task,              # 指定任务结构函数
        scoring_metrics=scoring_metrics,   # 指定评估指标
        experiment_config=experiment_config,  # 模型配置信息
    )


if __name__ == "__main__":
    main()  # 程序入口：执行主评估逻辑
