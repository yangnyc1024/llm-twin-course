from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录路径：
# __file__ 代表当前文件的路径，比如：/path/to/src/config/settings.py
# parent.parent.parent 意思是往上返回三层目录，比如从 /src/config/settings.py 到项目根目录
ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    """
    Settings 类：用于集中管理项目的配置（如数据库连接、账号密码等）。

    它继承自 Pydantic 的 BaseSettings，可以自动从环境变量 (.env 文件 或 系统环境变量) 中读取配置。
    """

    # model_config 是 pydantic-settings 的新特性，用于指定配置加载方式
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR,          # 指定环境变量文件的路径
        env_file_encoding="utf-8"   # 指定读取文件的编码格式
    )

    # MongoDB 数据库连接字符串
    MONGO_DATABASE_HOST: str = (
        "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    )

    # MongoDB 数据库名称
    MONGO_DATABASE_NAME: str = "twin"

    # 可选的 LinkedIn 登录凭证（如果需要爬取个人资料）
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None


# 实例化配置类（会自动读取默认值或环境变量）
settings = Settings()
