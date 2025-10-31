from abc import ABC, abstractmethod
from pydantic import BaseModel


class DataModel(BaseModel):
    """
    抽象的数据模型基类 (Abstract Base Data Model)

    用于所有类型的数据模型的基础定义。
    它继承自 Pydantic 的 BaseModel，
    因此可以自动进行数据验证和序列化（如 dict(), json() 等）。
    """

    entry_id: str  # 唯一标识数据条目的 ID（字符串形式）
    type: str      # 数据类型（例如 "post"、"article"、"repository" 等）


class VectorDBDataModel(ABC, DataModel):
    """
    抽象的向量数据库数据模型基类 (Abstract Vector Database Data Model)

    用于定义所有需要存入“向量数据库”（如 Qdrant、Pinecone、Milvus 等）的数据对象。
    继承自：
        - ABC：表示这是一个抽象基类（Abstract Base Class），不能被直接实例化；
        - DataModel：继承其基本字段和验证逻辑。

    子类（例如 EmbeddedPost、EmbeddedArticle 等）需要实现具体的向量化方法。
    """

    entry_id: int  # 这里重写为 int 类型，因为在 Qdrant 等向量数据库中，ID 通常为整数
    type: str      # 数据类型，同上

    @abstractmethod
    def to_payload(self) -> tuple:
        """
        抽象方法：将数据转换为可写入向量数据库的格式。
        每个子类必须实现此方法。

        通常返回一个包含以下内容的元组，例如：
            (id, vector, payload)
        其中：
            - id: 唯一标识（整数）
            - vector: 对象的嵌入向量（list[float]）
            - payload: 附加的元数据（dict）
        """
        pass
