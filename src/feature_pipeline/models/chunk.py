from typing import Optional
from models.base import DataModel


class PostChunkModel(DataModel):
    """
    ✅ 社交媒体帖子文本切片数据模型
    用于保存从 LinkedIn、Twitter 等平台抓取并分片（chunk）后的帖子数据。

    继承自 DataModel（基础模型），提供数据验证与统一结构。
    """

    entry_id: str             # 原始帖子（post）的唯一标识符
    platform: str             # 数据来源平台，例如 "linkedin"、"twitter"
    chunk_id: str             # 当前分片的唯一 ID（同一篇帖子可能被分成多个 chunk）
    chunk_content: str        # 分片后的文本内容（用于 embedding）
    author_id: str            # 帖子作者 ID
    image: Optional[str] = None  # （可选）帖子的配图 URL
    type: str                 # 数据类型（固定为 "post" 或类似标识）
    


class ArticleChunkModel(DataModel):
    """
    ✅ 文章文本切片数据模型
    用于保存从网站或 Medium、博客等来源提取的文章数据（分片后）。

    同样继承自 DataModel。
    """

    entry_id: str             # 原始文章 ID
    platform: str             # 来源平台（如 "medium"、"blog"）
    link: str                 # 文章链接 URL
    chunk_id: str             # 当前分片的唯一 ID
    chunk_content: str        # 分片的纯文本内容
    author_id: str            # 作者 ID
    type: str                 # 数据类型（如 "article"）



class RepositoryChunkModel(DataModel):
    """
    ✅ GitHub 仓库文本切片数据模型
    用于保存从 GitHub 仓库（或其他代码托管源）中提取的文档内容分片。

    例如 README、代码注释或描述文件的内容。
    """

    entry_id: str             # 原始仓库 ID
    name: str                 # 仓库名称
    link: str                 # 仓库链接
    chunk_id: str             # 分片 ID（一个仓库可能包含多个 chunk）
    chunk_content: str        # 代码或文档片段内容
    owner_id: str             # 仓库所有者 ID
    type: str                 # 数据类型（如 "repository"）
