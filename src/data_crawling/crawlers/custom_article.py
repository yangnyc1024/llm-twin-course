from urllib.parse import urlparse  # 用于解析 URL (提取域名、路径等)

from aws_lambda_powertools import Logger  # AWS Lambda 的日志工具（也可以本地使用）
from core.db.documents import ArticleDocument  # 数据库文档模型，用于保存文章信息
from langchain_community.document_loaders import AsyncHtmlLoader  # 异步网页加载器（LangChain 社区模块）
from langchain_community.document_transformers.html2text import Html2TextTransformer  # HTML 转纯文本的转换器

from .base import BaseCrawler  # 继承自基础爬虫框架

# 初始化日志工具，用于输出抓取过程信息
logger = Logger(service="llm-twin-course/crawler")


class CustomArticleCrawler(BaseCrawler):
    """
    自定义文章爬虫类。
    用于抓取网页文章（通用网页），并将提取的内容保存到数据库。
    """

    # 指定要使用的数据库模型，用于保存爬取结果
    model = ArticleDocument

    def __init__(self) -> None:
        # 初始化父类（BaseCrawler），虽然父类是抽象的，但为了保持一致性，这里还是显式调用
        super().__init__()

    def extract(self, link: str, **kwargs) -> None:
        """
        实现 BaseCrawler 的抽象方法。
        用于从指定网页链接中抓取文章内容，并保存到数据库中。

        :param link: 要爬取的文章链接
        :param kwargs: 额外参数（如 user ID）
        """

        # Step 1️⃣ 检查该链接是否已经在数据库中存在
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return  # 如果已存在，则跳过，避免重复抓取

        logger.info(f"Starting scrapping article: {link}")

        # Step 2️⃣ 使用 LangChain 的异步 HTML 加载器抓取网页
        loader = AsyncHtmlLoader([link])
        docs = loader.load()  # 加载网页内容（返回 LangChain Document 对象）

        # Step 3️⃣ 将 HTML 转为纯文本
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        doc_transformed = docs_transformed[0]  # 这里只取第一篇文章（因为只传了一个链接）

        # Step 4️⃣ 提取文章元信息和正文
        content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,  # 提取的纯文本正文
            "language": doc_transformed.metadata.get("language"),
        }

        # Step 5️⃣ 从 URL 中提取平台信息，例如 "medium.com" 或 "substack.com"
        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        # Step 6️⃣ 构建数据库文档实例
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=kwargs.get("user"),  # 作者ID（由调用方传入）
        )

        # Step 7️⃣ 保存到数据库
        instance.save()

        logger.info(f"Finished scrapping custom article: {link}")
