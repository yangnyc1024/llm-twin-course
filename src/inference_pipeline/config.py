from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录路径（当前文件的上上上级目录）
ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    # 配置 pydantic 的 Settings 模型，指定环境变量文件路径和编码格式
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # ===== Embeddings 模型配置 =====
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"  # 使用的向量嵌入模型名称
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512  # 单次可处理的最大输入长度
    EMBEDDING_SIZE: int = 384  # 向量维度大小
    EMBEDDING_MODEL_DEVICE: str = "cpu"  # 模型运行设备（可改为 "cuda"）

    # ===== OpenAI 模型配置 =====
    OPENAI_MODEL_ID: str = "gpt-4o-mini"  # 使用的 OpenAI 模型 ID
    OPENAI_API_KEY: str | None = None  # OpenAI API Key，从环境变量或 .env 文件加载

    # ===== Qdrant 向量数据库配置 =====
    QDRANT_DATABASE_HOST: str = "localhost"  # Qdrant 主机名，本地或 Docker 内为 "qdrant"
    QDRANT_DATABASE_PORT: int = 6333  # Qdrant 服务端口

    USE_QDRANT_CLOUD: bool = (
        False  # 若为 True，则启用云端 Qdrant 并需填写 QDRANT_CLOUD_URL 与 QDRANT_APIKEY
    )
    QDRANT_CLOUD_URL: str = "str"  # Qdrant Cloud 服务 URL
    QDRANT_APIKEY: str | None = None  # Qdrant Cloud API 密钥

    # ===== RAG（检索增强生成）配置 =====
    TOP_K: int = 5  # 从向量数据库检索的候选数量
    KEEP_TOP_K: int = 5  # 保留的最终候选数
    EXPAND_N_QUERY: int = 5  # 查询扩展的数量（增强检索效果）

    # ===== CometML 实验追踪配置 =====
    COMET_API_KEY: str  # CometML 的 API Key
    COMET_WORKSPACE: str  # CometML 工作空间名称
    COMET_PROJECT: str = "llm-twin"  # 默认项目名

    # ===== LLM 模型部署配置 =====
    HUGGINGFACE_ACCESS_TOKEN: str | None = None  # Hugging Face 访问令牌
    MODEL_ID: str = "pauliusztin/LLMTwin-Llama-3.1-8B"  # 自定义或微调后的模型 ID
    DEPLOYMENT_ENDPOINT_NAME: str = "twin"  # 部署的终端名称（用于推理接口）

    MAX_INPUT_TOKENS: int = 1536  # 最大输入 token 数
    MAX_TOTAL_TOKENS: int = 2048  # 输入 + 生成的总 token 限制
    MAX_BATCH_TOTAL_TOKENS: int = 2048  # 批量推理时的最大 token 限制

    # ===== AWS 认证配置 =====
    AWS_REGION: str = "eu-central-1"  # AWS 区域
    AWS_ACCESS_KEY: str | None = None  # AWS Access Key ID
    AWS_SECRET_KEY: str | None = None  # AWS Secret Access Key
    AWS_ARN_ROLE: str | None = None  # AWS IAM 角色 ARN（若需角色切换）


# 实例化 Settings 类，从 .env 文件或系统环境变量中加载配置
settings = Settings()
