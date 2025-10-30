from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取当前文件所在路径的上三级目录路径（作为项目根目录 ROOT_DIR）
# 举例：如果文件路径是 /app/src/config/settings.py
# 那么 ROOT_DIR = /app
ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    """
    用于集中管理项目配置项（例如 MongoDB、RabbitMQ 等连接信息）。
    继承自 Pydantic 的 BaseSettings，支持从 .env 文件或系统环境变量加载配置。
    """

    # 指定配置模型的元数据
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR,          # 指定 .env 文件路径（此处指项目根目录下）
        env_file_encoding="utf-8",  # 读取 .env 文件时使用 UTF-8 编码
    )

    # =====================
    # MongoDB 配置项
    # =====================
    # MongoDB 连接字符串，包含 3 个节点组成的副本集（replica set）
    # 用于保证 Mongo 集群的高可用性
    MONGO_DATABASE_HOST: str = (
        "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    )

    # MongoDB 数据库名称
    MONGO_DATABASE_NAME: str = "twin"

    # =====================
    # RabbitMQ 配置项
    # =====================
    # RabbitMQ 主机名（在 Docker 内部网络中通常是容器名“mq”，
    # 如果在本地运行则可改为 localhost）
    RABBITMQ_HOST: str = "mq"

    # RabbitMQ 端口号（默认 5672）
    RABBITMQ_PORT: int = 5672

    # RabbitMQ 默认用户名和密码（guest/guest）
    RABBITMQ_DEFAULT_USERNAME: str = "guest"
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"

    # RabbitMQ 队列名称（程序会将数据推送到这个队列）
    RABBITMQ_QUEUE_NAME: str = "default"


# 初始化配置对象，全局可直接使用
settings = Settings()
