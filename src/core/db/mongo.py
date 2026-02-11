from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from core.config import settings
from core.logger_utils import get_logger

logger = get_logger(__file__)


class MongoDatabaseConnector:
    """Singleton class to connect to MongoDB database."""
    # 中文注释：使用单例模式封装 MongoDB 连接，确保整个应用只创建一个 MongoClient 实例

    # 中文注释：类级别私有变量，用于缓存 MongoClient 实例
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs):
        # 中文注释：重写 __new__ 实现单例；如果尚未创建实例，则进行初始化
        if cls._instance is None:
            try:
                # 中文注释：通过配置中的 MONGO_DATABASE_HOST 创建 MongoClient 连接
                cls._instance = MongoClient(settings.MONGO_DATABASE_HOST)
                logger.info(
                    f"Connection to database with uri: {settings.MONGO_DATABASE_HOST} successful"
                )
            except ConnectionFailure:
                # 中文注释：连接失败时记录错误日志并抛出异常
                logger.error(f"Couldn't connect to the database.")

                raise

        # 中文注释：返回已创建或已缓存的 MongoClient 实例
        return cls._instance

    def get_database(self):
        # 中文注释：获取具体数据库实例（基于配置中的数据库名称）
        assert self._instance, "Database connection not initialized"

        # 中文注释：返回指定名称的 database 对象（类似 client["db_name"]）
        return self._instance[settings.MONGO_DATABASE_NAME]

    def close(self):
        # 中文注释：关闭 MongoDB 连接（通常用于应用关闭阶段）
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")


# 中文注释：在模块加载时初始化单例连接对象，供其他模块直接 import 使用
connection = MongoDatabaseConnector()
