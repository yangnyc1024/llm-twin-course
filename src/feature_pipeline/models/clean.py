from typing import Optional, Tuple
from models.base import VectorDBDataModel


class PostCleanedModel(VectorDBDataModel):
    """
    ✅ 已清洗的社交媒体帖子数据模型
    用于保存从社交媒体（如 LinkedIn、Twitter）抓取并清洗后的帖子内容。
    继承自 VectorDBDataModel，因此必须实现 `to_payload()` 方法，
    用于将数据转换为可写入 Qdrant（或其他向量数据库）的格式。
    """

    entry_id: str                 # 帖子唯一 ID
    platform: str                 # 来源平台（如 "linkedin"、"twitter"）
    cleaned_content: str          # 清洗后的纯文本内容（去除 HTML、表情符号、多余空格等）
    author_id: str                # 帖子作者 ID
    image: Optional[str] = None   # 可选字段：配图 URL
    type: str                     # 数据类型（如 "posts"）

    def to_payload(self) -> Tuple[str, dict]:
        """
        将当前对象转换为向量数据库可接受的格式：
        返回一个 (entry_id, payload) 元组。

        - entry_id: 作为主键 ID（唯一标识）
        - payload: dict 形式的元数据内容
        """
        data = {
            "platform": self.platform,
            "author_id": self.author_id,
            "cleaned_content": self.cleaned_content,
            "image": self.image,
            "type": self.type,
        }

        return self.entry_id, data



class ArticleCleanedModel(VectorDBDataModel):
    """
    ✅ 已清洗的文章数据模型
    用于保存从 Medium、博客或新闻站点抓取的文章文本内容（经过清洗后）。
    """

    entry_id: str                 # 原始文章 ID
    platform: str                 # 来源平台（如 "medium"、"news"）
    link: str                     # 文章 URL 链接
    cleaned_content: str          # 清洗后的文章文本
    author_id: str                # 作者 ID
    type: str                     # 数据类型（如 "articles"）

    def to_payload(self) -> Tuple[str, dict]:
        """
        将文章数据转换为 (entry_id, payload) 格式。
        """
        data = {
            "platform": self.platform,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "author_id": self.author_id,
            "type": self.type,
        }

        return self.entry_id, data



class RepositoryCleanedModel(VectorDBDataModel):
    """
    ✅ 已清洗的代码仓库数据模型
    用于保存从 GitHub（或其他代码托管平台）抓取、清洗后的仓库文档内容。
    """

    entry_id: str                 # 仓库唯一 ID
    name: str                     # 仓库名称
    link: str                     # 仓库链接 URL
    cleaned_content: str          # 清洗后的仓库文本内容（例如 README 或代码注释）
    owner_id: str                 # 仓库所有者 ID
    type: str                     # 数据类型（如 "repositories"）

    def to_payload(self) -> Tuple[str, dict]:
        """
        将仓库数据转换为 (entry_id, payload) 格式。
        """
        data = {
            "name": self.name,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "owner_id": self.owner_id,
            "type": self.type,
        }

        return self.entry_id, data
