import json
import logging

from bson import json_util              # 用于处理 MongoDB 的 BSON 数据类型（例如 ObjectId）
from config import settings             # 导入配置文件（包含 RabbitMQ 的配置等）
from core.db.mongo import MongoDatabaseConnector  # 自定义的 MongoDB 数据库连接类
from core.logger_utils import get_logger          # 自定义日志工具函数
from core.mq import publish_to_rabbitmq           # 发布消息到 RabbitMQ 的函数

# 获取一个带文件名的 logger 实例
logger = get_logger(__file__)


def stream_process():
    """
    实时监听 MongoDB 数据库中的变化（特别是 insert 操作），
    并将新插入的数据通过 RabbitMQ 推送到消息队列。
    """
    try:
        # 1️⃣ 连接 MongoDB 数据库
        client = MongoDatabaseConnector()
        db = client["twin"]  # 指定数据库名称为 "twin"
        logging.info("Connected to MongoDB.")

        # 2️⃣ 开启 MongoDB 的 Change Stream，监听集合中的变化
        # 这里只监听 "insert" 操作类型，即有新文档插入时触发
        changes = db.watch([{"$match": {"operationType": {"$in": ["insert"]}}}])

        # 3️⃣ 遍历监听到的变化事件
        for change in changes:
            # 获取被修改（或插入）的集合名称
            data_type = change["ns"]["coll"]

            # 将 MongoDB 自动生成的 ObjectId 转换为字符串
            entry_id = str(change["fullDocument"]["_id"])

            # 从文档中移除 _id 字段（因为 ObjectId 不是标准 JSON）
            change["fullDocument"].pop("_id")

            # 新增字段 "type" 和 "entry_id"，标识数据来源类型和唯一 ID
            change["fullDocument"]["type"] = data_type
            change["fullDocument"]["entry_id"] = entry_id

            # 4️⃣ 限制只处理三种集合类型
            if data_type not in ["articles", "posts", "repositories"]:
                logging.info(f"Unsupported data type: '{data_type}'")
                continue  # 其他集合类型跳过

            # 5️⃣ 使用 bson.json_util 序列化文档，确保 MongoDB 特有类型可转成 JSON
            data = json.dumps(change["fullDocument"], default=json_util.default)
            logger.info(
                f"Change detected and serialized for a data sample of type {data_type}."
            )

            # 6️⃣ 将序列化后的数据发送到 RabbitMQ 队列中
            publish_to_rabbitmq(queue_name=settings.RABBITMQ_QUEUE_NAME, data=data)
            logger.info(f"Data of type '{data_type}' published to RabbitMQ.")

    except Exception as e:
        # 7️⃣ 捕获所有异常并打印日志
        logger.error(f"An error occurred: {e}")


# 8️⃣ 如果该脚本是主程序运行，则执行流处理函数
if __name__ == "__main__":
    stream_process()

