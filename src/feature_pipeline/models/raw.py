from typing import Optional
from models.base import DataModel


class RepositoryRawModel(DataModel):
    """
    ✅ 原始仓库数据模型（Raw Repository Model）
    用于保存从 GitHub（或其他代码托管平台）爬取的原始仓库内容。

    继承自 DataModel（统一基础结构），
    是清洗前的最原始数据结构。
    """

    name: str            # 仓库名称，例如 "llm-twin-course"
    link: str            # 仓库链接 URL，例如 "https://github.com/decodingai-magazine/llm-twin-course"
    content: dict        # 仓库内容，以字典形式存储（可能包含 README、代码、文件树等）
    owner_id: str        # 仓库所有者 ID
    


class ArticleRawModel(DataModel):
    """
    ✅ 原始文章数据模型（Raw Article Model）
    用于保存从网站（如 Medium、Blog、News site）爬取的文章原始内容。

    该模型通常包含网页 HTML 或半结构化的文本内容，
    后续会经过文本提取与清洗。
    """

    platform: str        # 来源平台（如 "medium"、"blog"、"news"）
    link: str            # 文章链接 URL
    content: dict        # 原始网页内容（可能是 HTML 文本、metadata、段落列表等）
    author_id: str       # 作者 ID
    


class PostsRawModel(DataModel):
    """
    ✅ 原始社交媒体帖子数据模型（Raw Post Model）
    用于保存从 LinkedIn、Twitter、Reddit 等社交媒体爬取的原始帖子内容。

    通常包含用户信息、帖子正文、图片等。
    """

    platform: str                  # 来源平台（如 "linkedin"、"twitter"）
    content: dict                  # 原始帖子内容（例如 JSON 格式，包含文本、发布时间、meta 等）
    author_id: str | None = None   # 帖子作者 ID，可选（部分平台可能匿名或未提供）
    image: Optional[str] = None    # 帖子图片（URL，可选）
