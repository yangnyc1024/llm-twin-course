from abc import ABC, abstractmethod

from models.base import DataModel
from models.clean import ArticleCleanedModel, PostCleanedModel, RepositoryCleanedModel
from models.raw import ArticleRawModel, PostsRawModel, RepositoryRawModel
from utils.cleaning import clean_text


class CleaningDataHandler(ABC):
    """
    抽象基类：定义“数据清洗处理器（Cleaning Handler）”的统一接口。
    
    用途：
    - 在原始数据(raw data)进入模型/数据库之前，通常需要清洗（去HTML、去标签、合并段落、去噪声）。
    - 不同类型（帖子、文章、代码仓库）的数据结构不同，因此需要专门的清洗逻辑。
    - 抽象类通过定义抽象方法 `clean(...)` 统一约束接口，便于扩展。
    """

    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        """
        抽象方法：接收一个“原始数据模型 (RawModel)”，返回一个“清洗后的数据模型 (CleanedModel)”。
        """
        pass


class PostCleaningHandler(CleaningDataHandler):
    """
    清洗社交媒体帖子(Post) 的数据。
    
    输入：PostsRawModel（原始帖子数据，可能含有HTML、表情符号等噪声）
    输出：PostCleanedModel（清洗后的纯文本内容）
    """

    def clean(self, data_model: PostsRawModel) -> PostCleanedModel:
        # 1️⃣ 拼接原始内容字段（通常 content 是一个 dict，如 {"text": "...", "caption": "..."}）
        joined_text = (
            "".join(data_model.content.values()) if data_model and data_model.content else None
        )

        # 2️⃣ 调用通用清洗函数 clean_text 去除噪声（HTML 标签、特殊字符、换行符等）
        return PostCleanedModel(
            entry_id=data_model.entry_id,                    # 原始记录唯一标识
            platform=data_model.platform,                    # 来源平台（linkedin、twitter等）
            cleaned_content=clean_text(joined_text),          # 清洗后文本
            author_id=data_model.author_id,                  # 作者信息
            image=data_model.image if data_model.image else None,  # 图片（如果有）
            type=data_model.type,                            # 类型标识
        )


class ArticleCleaningHandler(CleaningDataHandler):
    """
    清洗文章（Article）的数据。
    
    输入：ArticleRawModel（原始文章数据）
    输出：ArticleCleanedModel（清洗后文本）
    """

    def clean(self, data_model: ArticleRawModel) -> ArticleCleanedModel:
        joined_text = (
            "".join(data_model.content.values()) if data_model and data_model.content else None
        )

        return ArticleCleanedModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            link=data_model.link,                            # 原文链接
            cleaned_content=clean_text(joined_text),
            author_id=data_model.author_id,
            type=data_model.type,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    """
    清洗代码仓库（Repository）的数据。
    
    输入：RepositoryRawModel（原始仓库数据，可能由README、代码注释、文件内容拼接而成）
    输出：RepositoryCleanedModel（清洗后的仓库文本）
    """

    def clean(self, data_model: RepositoryRawModel) -> RepositoryCleanedModel:
        joined_text = (
            "".join(data_model.content.values()) if data_model and data_model.content else None
        )

        return RepositoryCleanedModel(
            entry_id=data_model.entry_id,
            name=data_model.name,                             # 仓库名称
            link=data_model.link,                             # 仓库链接
            cleaned_content=clean_text(joined_text),
            owner_id=data_model.owner_id,                     # 仓库所有者
            type=data_model.type,
        )


# 🧩 文件整体逻辑说明
# 1️⃣ 模块定位
# 这个文件属于整个 data logic / ETL 管道 的第二阶段：
# Raw -> Clean -> Chunk -> Embed -> Upsert
# 即在「原始数据」被爬取并存入数据库后，进入清洗阶段，去除噪声并统一格式。

# 2️⃣ 功能要点
# 类名	输入模型	输出模型	主要清洗任务
# PostCleaningHandler	PostsRawModel	PostCleanedModel	拼接内容字段、去HTML/噪声
# ArticleCleaningHandler	ArticleRawModel	ArticleCleanedModel	清洗正文内容、保留原始链接
# RepositoryCleaningHandler	RepositoryRawModel	RepositoryCleanedModel	合并多文件内容、清理注释、README 等文本

# 3️⃣ 通用逻辑
# 所有 handler 继承 CleaningDataHandler；
# 调用 clean_text() 执行通用清洗（该函数在 utils.cleaning 中定义）；
# 输出的 CleanedModel 会在下一个阶段由 ChunkingDataHandler 进行文本分块。

# 4️⃣ 工程设计思想

# 面向抽象：使用抽象基类(ABC)约束清洗接口；
# 职责单一：每个类只处理一种数据类型；
# 解耦性好：不负责存储/日志/embedding，只专注于数据转换；
# 可扩展：未来可以轻松添加新的 handler（例如 PDFCleaningHandler 或 CodeFileCleaningHandler）。