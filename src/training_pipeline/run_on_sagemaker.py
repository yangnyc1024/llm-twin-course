import argparse
import sys
from pathlib import Path

# 为了在本地开发或教学环境中方便导入多模块（如 'core'、'feature_pipeline'），
# 我们手动将 './src' 目录添加到 Python 的搜索路径 (PYTHONPATH)。
# ⚠️ 注意：此方法仅用于开发/教学场景，不建议用于生产环境。
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
from huggingface_hub import HfApi
from sagemaker.huggingface import HuggingFace

# 初始化日志记录器（方便输出运行信息）
logger = logger_utils.get_logger(__file__)

from config import settings

# 获取当前 finetuning 目录路径（即包含本文件的路径）
finetuning_dir = Path(__file__).resolve().parent
# 读取微调所需依赖的 requirements.txt 文件路径
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    is_dummy: bool = False,
) -> None:
    """
    在 AWS SageMaker 上运行 Hugging Face 模型微调任务。
    参数：
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每个设备的 batch 大小
        learning_rate: 学习率
        is_dummy: 是否启用 Dummy 模式（即快速测试）
    """
    # ---------------------- 参数合法性检查 ----------------------
    # 确保环境配置中包含必要的访问凭证与设置，否则直接抛出异常提示用户更新 .env 文件
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token (HUGGINGFACE_ACCESS_TOKEN) is required. Update your .env file."
    assert (
        settings.AWS_ARN_ROLE
    ), "AWS ARN role (AWS_ARN_ROLE) is required. Update your .env file."
    assert (
        settings.COMET_API_KEY
    ), "Comet ML API key (COMET_API_KEY) is required. Update your .env file."
    assert (
        settings.COMET_WORKSPACE
    ), "Comet ML workspace (COMET_WORKSPACE) is required. Update your .env file."
    assert (
        settings.COMET_PROJECT
    ), "Comet ML project name (COMET_PROJECT) is required. Update your .env file."

    # ---------------------- 路径检查 ----------------------
    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(
            f"The file {finetuning_requirements_path} does not exist."
        )

    # ---------------------- Hugging Face 用户身份验证 ----------------------
    api = HfApi()
    # 调用 Hugging Face Hub API 验证当前用户身份
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # ---------------------- 定义训练超参数 ----------------------
    # 这些参数会传递给 SageMaker 容器内的 `finetune.py` 脚本
    hyperparameters = {
        "base_model_name": settings.HUGGINGFACE_BASE_MODEL_ID,  # 基础模型（如 Llama 3.1 8B）
        "dataset_id": settings.DATASET_ID,                      # 数据集 ID（Comet artifact 名称）
        "num_train_epochs": num_train_epochs,                   # 训练轮数
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "model_output_huggingface_workspace": huggingface_user,  # 训练后模型保存的 HF Workspace
    }
    # 若启用 Dummy 模式（仅部分数据、快速测试）
    if is_dummy:
        hyperparameters["is_dummy"] = True

    # ---------------------- 创建 HuggingFace SageMaker Estimator ----------------------
    # HuggingFace Estimator 是 SageMaker 官方封装的训练作业启动类
    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",              # SageMaker 容器内的入口脚本
        source_dir=str(finetuning_dir),         # 上传到 SageMaker 的源码目录
        instance_type="ml.g5.2xlarge",          # 使用 GPU 实例 (g5.2xlarge: NVIDIA A10G)
        instance_count=1,                       # 单节点训练
        role=settings.AWS_ARN_ROLE,             # AWS 执行角色（带有 SageMaker 权限）
        transformers_version="4.36",            # 对应的 Transformers 版本
        pytorch_version="2.1",                  # 对应的 PyTorch 版本
        py_version="py310",                     # Python 版本
        hyperparameters=hyperparameters,        # 传递训练超参数
        requirements_file=finetuning_requirements_path,  # 安装依赖文件
        environment={                           # 运行环境变量
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_WORKSPACE": settings.COMET_WORKSPACE,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    # ---------------------- 启动 SageMaker 训练作业 ----------------------
    huggingface_estimator.fit()


if __name__ == "__main__":
    # 命令行解析器：支持通过 --is-dummy 参数启用 Dummy 模式
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-dummy", action="store_true", help="Run in dummy mode")
    args = parser.parse_args()

    logger.info(f"Is the training pipeline in DUMMY mode? '{args.is_dummy}'")

    # 调用主函数执行微调任务
    run_finetuning_on_sagemaker(is_dummy=args.is_dummy)
