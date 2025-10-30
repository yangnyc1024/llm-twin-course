import json
import time
from datetime import datetime
from typing import Generic, Iterable, List, Optional, TypeVar

from bytewax.inputs import FixedPartitionedSource, StatefulSourcePartition
from config import settings
from core import get_logger
from core.mq import RabbitMQConnection

logger = get_logger(__name__)

# 泛型类型，用于定义通用的数据类型
DataT = TypeVar("DataT")
MessageT = TypeVar("MessageT")


class RabbitMQPartition(StatefulSourcePartition, Generic[DataT, MessageT]):
    """
    🚀 Bytewax 与 RabbitMQ 之间的桥接类。
    负责从 RabbitMQ 队列中拉取消息，并提供给 Bytewax streaming pipeline。

    继承自 StatefulSourcePartition：
    - 支持状态快照(snapshot)，可以在崩溃/重启时恢复处理进度。
    """

    def __init__(self, queue_name: str, resume_state: MessageT | None = None) -> None:
        # 用于存储正在处理但尚未确认（ack）的消息 ID
        self._in_flight_msg_ids = resume_state or set()
        self.queue_name = queue_name

        # 建立与 RabbitMQ 的连接
        self.connection = RabbitMQConnection()
        self.connection.connect()
        self.channel = self.connection.get_channel()

    def next_batch(self, sched: Optional[datetime]) -> Iterable[DataT]:
        """
        🧩 Bytewax 每次调用此方法来“拉取一批新消息”。
        - 如果队列中有新消息，则取出并返回（转换为 Python 对象）。
        - 如果没有消息，则返回空列表。
        - 如果连接失败，则重连并等待重试。
        """
        try:
            # 从 RabbitMQ 队列中拉取一条消息（非阻塞）
            method_frame, header_frame, body = self.channel.basic_get(
                queue=self.queue_name,
                auto_ack=True  # 自动确认（简化模式）
            )
        except Exception:
            logger.error(
                f"Error while fetching message from queue.", queue_name=self.queue_name
            )
            time.sleep(10)  # 如果失败，等待 10 秒后重试

            # 尝试重连 RabbitMQ
            self.connection.connect()
            self.channel = self.connection.get_channel()

            return []

        # 如果成功取到消息
        if method_frame:
            message_id = method_frame.delivery_tag
            self._in_flight_msg_ids.add(message_id)

            # 将 JSON 格式的消息体反序列化为 Python 字典
            return [json.loads(body)]
        else:
            # 如果队列为空
            return []

    def snapshot(self) -> MessageT:
        """
        🧷 保存当前处理进度的状态（即未确认消息 ID 列表）。
        用于 Bytewax 的 checkpoint 机制。
        """
        return self._in_flight_msg_ids

    def garbage_collect(self, state):
        """
        🧹 清理已确认（处理完成）的消息。
        通常在 Bytewax 完成一个 batch 的处理后调用。
        """
        closed_in_flight_msg_ids = state
        for msg_id in closed_in_flight_msg_ids:
            # 向 RabbitMQ 发送 ACK 确认该消息已被成功处理
            self.channel.basic_ack(delivery_tag=msg_id)
            self._in_flight_msg_ids.remove(msg_id)

    def close(self):
        """
        📴 关闭连接通道。
        通常在 pipeline 结束或退出时调用。
        """
        self.channel.close()


class RabbitMQSource(FixedPartitionedSource):
    """
    📦 定义一个固定分区（FixedPartitionedSource）的数据源。
    由于 RabbitMQ 本身是单队列模型，这里只定义一个“单分区”。
    """

    def list_parts(self) -> List[str]:
        # Bytewax 要求返回分区列表，这里返回一个固定的分区
        return ["single partition"]

    def build_part(
        self, now: datetime, for_part: str, resume_state: MessageT | None = None
    ) -> StatefulSourcePartition[DataT, MessageT]:
        """
        🔧 构建对应分区的读取逻辑，即返回一个 RabbitMQPartition 实例。
        每个分区都会有自己的连接和状态。
        """
        return RabbitMQPartition(queue_name=settings.RABBITMQ_QUEUE_NAME)

# 爬虫 / 数据采集器
#         │
#         ▼
#      RabbitMQ 队列
#         │
#         ▼
#  [RabbitMQSource (Bytewax Input)]
#         │
#         ▼
#  Bytewax Pipeline (stream processing)
#         │
#         ├── 清洗 / 特征抽取 (feature_engineering)
#         └── 输出到 MongoDB / Qdrant / Feature Store
