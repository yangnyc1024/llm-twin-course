from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录路径：
# Path(__file__) 是当前文件路径，例如：/project/core/config/settings.py
# .parent.parent.parent.parent 表示往上返回四层目录
# 将 Path 转成字符串，方便之后拼接路径或加载 .env 文件
ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)


class Settings(BaseSettings):
    """
    统一的项目配置类（基于 Pydantic Settings 管理所有环境变量和配置参数）

    - 自动从环境变量或 `.env` 文件中加载配置
    - 支持类型校验和默认值
    - 用于集中管理 HuggingFace、Comet、AWS 等外部服务的连接配置
    """

    # 指定配置模型信息（告诉 Pydantic 从哪加载 .env 文件）
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # ---------- Hugging Face 配置 ----------
    # 使用的基础模型 ID（这里默认是 Llama 3.1 8B 模型）
    HUGGINGFACE_BASE_MODEL_ID: str = "meta-llama/Llama-3.1-8B"
    # Hugging Face 访问令牌（如果需要访问私有模型或上传）
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # ---------- Comet ML 配置 ----------
    # Comet API 密钥，用于实验追踪
    COMET_API_KEY: str | None = None
    # Comet Workspace 名称
    COMET_WORKSPACE: str | None = None
    # 默认项目名称（在 Comet 中实验将归入此项目）
    COMET_PROJECT: str = "llm-twin"

    # Comet artifact 中的数据集 ID
    # 即之前上传的 fine-tuning instruct 数据集的唯一标识
    DATASET_ID: str = "articles-instruct-dataset"

    # ---------- AWS 配置 ----------
    # 所使用的 AWS 区域（默认欧洲中部）
    AWS_REGION: str = "eu-central-1"
    # AWS 访问密钥 ID
    AWS_ACCESS_KEY: str | None = None
    # AWS 密钥（secret access key）
    AWS_SECRET_KEY: str | None = None
    # AWS 角色 ARN（通常用于 SageMaker 或 EC2 访问控制）
    AWS_ARN_ROLE: str | None = None


# 创建 Settings 实例，加载环境配置
# 一旦初始化完成，可以通过 settings.HUGGINGFACE_BASE_MODEL_ID 等属性访问配置值
settings = Settings()
