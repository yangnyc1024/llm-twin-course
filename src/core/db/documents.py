import uuid
from typing import List, Optional

from pydantic import UUID4, BaseModel, ConfigDict, Field
from pymongo import errors

import core.logger_utils as logger_utils
from core.db.mongo import connection
from core.errors import ImproperlyConfigured

_database = connection.get_database("twin")
logger = logger_utils.get_logger(__name__)


class BaseDocument(BaseModel):
    # 中文注释：文档主键 ID（UUID4），默认自动生成；用于业务侧统一使用 `id`
    id: UUID4 = Field(default_factory=uuid.uuid4)

    # 中文注释：
    # - from_attributes=True: 支持从对象属性构建（例如 ORM/对象实例）
    # - populate_by_name=True: 允许使用字段名/别名进行赋值与解析
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @classmethod
    def from_mongo(cls, data: dict):
        """Convert "_id" (str object) into "id" (UUID object)."""
        # 中文注释：从 MongoDB 读出的原始字典为空时，直接返回
        if not data:
            return data

        # 中文注释：MongoDB 使用 `_id` 作为主键字段，这里取出并映射到业务字段 `id`
        id = data.pop("_id", None)
        # 中文注释：将剩余字段 + id 一起传入 Pydantic 模型构造
        return cls(**dict(data, id=id))

    def to_mongo(self, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        # 中文注释：是否只导出被显式设置过的字段（Pydantic model_dump 参数）
        exclude_unset = kwargs.pop("exclude_unset", False)
        # 中文注释：是否使用字段别名导出（例如 author_id/owner_id 等）
        by_alias = kwargs.pop("by_alias", True)

        # 中文注释：将 Pydantic 模型序列化为 dict，作为 MongoDB 写入用的 payload
        parsed = self.model_dump(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )

        # 中文注释：若 `_id` 不存在但 `id` 存在，则把 `id` 转成字符串写入 `_id`
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        return parsed

    def save(self, **kwargs):
        # 中文注释：根据子类定义的 collection 名称获取 MongoDB collection
        collection = _database[self._get_collection_name()]

        try:
            # 中文注释：插入单条文档；insert_one 返回 InsertOneResult
            result = collection.insert_one(self.to_mongo(**kwargs))
            # 中文注释：返回 MongoDB 插入后的 _id（这里是字符串形式的 UUID）
            return result.inserted_id
        except errors.WriteError:
            # 中文注释：捕获写入错误并打印堆栈，返回 None 表示失败
            logger.exception("Failed to insert document.")

            return None

    @classmethod
    def get_or_create(cls, **filter_options) -> Optional[str]:
        # 中文注释：通过过滤条件查找；不存在则创建并插入；返回 id（字符串）或 None
        collection = _database[cls._get_collection_name()]
        try:
            # 中文注释：先尝试查找是否已有匹配文档
            instance = collection.find_one(filter_options)
            if instance:
                # 中文注释：若存在，转换成模型并返回其 id（UUID -> str）
                return str(cls.from_mongo(instance).id)

            # 中文注释：不存在则用过滤条件构造新实例（要求这些字段足以构造模型）
            new_instance = cls(**filter_options)
            # 中文注释：写入 MongoDB，save() 返回 inserted_id 或 None
            new_instance = new_instance.save()
            return new_instance
        except errors.OperationFailure:
            # 中文注释：MongoDB 操作失败（权限/命令/连接等），返回 None
            logger.exception("Failed to retrieve or create document.")

            return None

    @classmethod
    def find(cls, **filter_options):
        # 中文注释：按条件查询单条；返回 Pydantic 模型实例或 None
        collection = _database[cls._get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                # 中文注释：MongoDB 文档 -> Pydantic 模型
                return cls.from_mongo(instance)

            return None
        except errors.OperationFailure:
            # 中文注释：查询失败时记录错误并返回 None
            logger.error("Failed to retrieve document")

            return None

    @classmethod
    def bulk_insert(cls, documents: List, **kwargs) -> Optional[List[str]]:
        # 中文注释：批量插入文档列表；返回 inserted_ids 或 None
        collection = _database[cls._get_collection_name()]
        try:
            # 中文注释：将每个文档转为 MongoDB dict 形式后 insert_many
            result = collection.insert_many(
                [doc.to_mongo(**kwargs) for doc in documents]
            )
            return result.inserted_ids
        except errors.WriteError:
            # 中文注释：批量写入失败时记录堆栈并返回 None
            logger.exception("Failed to insert documents.")

            return None

    @classmethod
    def _get_collection_name(cls):
        # 中文注释：要求子类提供 Settings.name 作为 collection 名称
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        # 中文注释：返回 MongoDB collection 名称字符串
        return cls.Settings.name


class UserDocument(BaseDocument):
    # 中文注释：用户的名
    first_name: str
    # 中文注释：用户的姓
    last_name: str

    class Settings:
        # 中文注释：MongoDB collection 名称
        name = "users"


class RepositoryDocument(BaseDocument):
    # 中文注释：仓库名称
    name: str
    # 中文注释：仓库链接（例如 GitHub URL）
    link: str
    # 中文注释：仓库内容/元数据（结构化字典，具体结构由上游定义）
    content: dict
    # 中文注释：仓库所属用户 ID；使用 alias 保持与数据库字段一致
    owner_id: str = Field(alias="owner_id")

    class Settings:
        # 中文注释：MongoDB collection 名称
        name = "repositories"


class PostDocument(BaseDocument):
    # 中文注释：发布平台（例如 twitter/linkedin 等）
    platform: str
    # 中文注释：帖子内容（结构化字典）
    content: dict
    # 中文注释：作者 ID；使用 alias 保持与数据库字段一致
    author_id: str = Field(alias="author_id")

    class Settings:
        # 中文注释：MongoDB collection 名称
        name = "posts"


# 中文注释：文章文档（与 Post 类似，但包含 link 字段，且 collection 为 articles）
class ArticleDocument(BaseDocument):
    # 中文注释：文章平台（例如 medium/substack 等）
    platform: str
    # 中文注释：文章链接（URL）
    link: str
    # 中文注释：文章内容（结构化字典）
    content: dict
    # 中文注释：作者 ID；使用 alias 保持与数据库字段一致
    author_id: str = Field(alias="author_id")

    class Settings:
        # 中文注释：MongoDB collection 名称
        name = "articles"
