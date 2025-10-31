import hashlib
from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import ArticleChunkModel, PostChunkModel, RepositoryChunkModel
from models.clean import ArticleCleanedModel, PostCleanedModel, RepositoryCleanedModel
from utils.chunking import chunk_text


class ChunkingDataHandler(ABC):
    """
    抽象基类：定义“切分（chunking）数据处理器”的统一接口。

    设计动机：
    - 在清洗后的文本进入向量化/入库之前，通常需要先进行文本分块（chunking）。
    - 不同数据类型（帖子、文章、代码仓库）的字段结构不同，因此切分后的数据模型也不同。
    - 通过抽象类统一约束 `chunk(...)` 方法签名，便于后续扩展新的数据类型时保持一致性。
    """

    @abstractmethod
    def chunk(self, data_model: DataModel) -> list[DataModel]:
        """
        将“清洗后的数据模型”切分为多个“分块后的数据模型”。

        参数：
            data_model: 任意实现了 DataModel 接口/基类的实例（具体子类在子处理器中限定）。
        返回：
            分块后的数据模型列表（每个元素代表一个 chunk）。
        """
        pass


class PostChunkingHandler(ChunkingDataHandler):
    """
    处理社交媒体帖子（Post）的文本分块。

    输入：PostCleanedModel（已清洗）
    输出：list[PostChunkModel]（分块结果）
    """

    def chunk(self, data_model: PostCleanedModel) -> list[PostChunkModel]:
        data_models_list: list[PostChunkModel] = []

        # 1) 取出清洗后的文本内容
        text_content = data_model.cleaned_content

        # 2) 调用通用的 chunk_text 工具函数，根据策略（如长度、句子边界等）进行文本切分
        chunks = chunk_text(text_content)

        # 3) 将每个 chunk 封装成 PostChunkModel
        for chunk in chunks:
            model = PostChunkModel(
                entry_id=data_model.entry_id,            # 原始记录的唯一标识，便于回溯
                platform=data_model.platform,            # 平台信息（如 linkedin / twitter 等）
                # 使用 md5(UTF-8 编码的 chunk 文本) 作为 chunk_id
                # 注：md5 碰撞概率虽低但非零；如需更稳妥可改为 sha256 或在哈希中加入 entry_id/salt。
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk,                     # 具体的分块文本
                author_id=data_model.author_id,          # 作者标识
                image=data_model.image if data_model.image else None,  # 可能附带的图片信息
                type=data_model.type,                    # 业务类型（与下游集合名/路由相关）
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    """
    处理文章（Article）的文本分块。

    输入：ArticleCleanedModel（已清洗）
    输出：list[ArticleChunkModel]（分块结果）
    """

    def chunk(self, data_model: ArticleCleanedModel) -> list[ArticleChunkModel]:
        data_models_list: list[ArticleChunkModel] = []

        # 1) 取出清洗后的正文
        text_content = data_model.cleaned_content

        # 2) 通用切分
        chunks = chunk_text(text_content)

        # 3) 封装为 ArticleChunkModel
        for chunk in chunks:
            model = ArticleChunkModel(
                entry_id=data_model.entry_id,            # 源记录 ID
                platform=data_model.platform,            # 数据来源平台（如 medium、自定义爬虫等）
                link=data_model.link,                    # 文章原始链接，便于追踪与展示
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk,
                author_id=data_model.author_id,
                type=data_model.type,
            )
            data_models_list.append(model)

        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler):
    """
    处理代码仓库（Repository）文本/内容的分块。

    输入：RepositoryCleanedModel（已清洗，通常包含仓库内聚合后的文本或抽取内容）
    输出：list[RepositoryChunkModel]
    """

    def chunk(self, data_model: RepositoryCleanedModel) -> list[RepositoryChunkModel]:
        data_models_list: list[RepositoryChunkModel] = []

        # 1) 仓库清洗后的合并文本（可能是 README、代码注释、文档等内容的整合）
        text_content = data_model.cleaned_content

        # 2) 切分
        chunks = chunk_text(text_content)

        # 3) 封装为 RepositoryChunkModel
        for chunk in chunks:
            model = RepositoryChunkModel(
                entry_id=data_model.entry_id,            # 源记录 ID
                name=data_model.name,                    # 仓库名
                link=data_model.link,                    # 仓库链接
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk,
                owner_id=data_model.owner_id,            # 仓库所有者 ID
                type=data_model.type,
            )
            data_models_list.append(model)

        return data_models_list


# ============================
# 设计&工程化建议（可选）
# ============================
# 1) chunk_id 生成：
#    - 为降低极端 md5 碰撞风险，建议：hashlib.sha256(f"{data_model.entry_id}:{chunk}".encode()).hexdigest()
#    - 或者附加序号：chunk_idx = enumerate(chunks)，将 {entry_id}-{idx} 作为 chunk_id。
#
# 2) chunk_text 策略：
#    - 根据 RAG/向量检索的需求调整：按字数、按句子、按段落、按 Token（结合 tiktoken）等。
#    - 返回结构可包含 metadata（如起止位置、token 数等）以便后续索引时更灵活。
#
# 3) 类型约束：
#    - 抽象基类签名使用较宽的 DataModel；在具体实现中通过类型提示收窄为具体的 *CleanedModel，
#      保持灵活性同时让 IDE/类型检查更友好。
#
# 4) 幂等性：
#    - 如果需要保证重复执行 chunk 不产生重复入库，可在下游以 (entry_id, chunk_id) 做唯一索引。
#
# 5) 上下游解耦：
#    - 当前处理器仅负责“从 CleanedModel -> ChunkModel”的转化，避免耦合存储/向量化逻辑，
#      让 pipeline 更清晰（clean -> chunk -> embed -> upsert）。
