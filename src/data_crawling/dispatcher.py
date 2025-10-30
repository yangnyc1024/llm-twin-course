from aws_lambda_powertools import Logger
from crawlers.base import BaseCrawler
from crawlers.custom_article import CustomArticleCrawler
import re

# 初始化日志记录器
# 这是 AWS Lambda Powertools 提供的高质量日志工具，支持结构化日志输出
# 方便在 AWS CloudWatch 等环境中查看和过滤日志
logger = Logger(service="llm-twin-course/crawler")


class CrawlerDispatcher:
    """
    爬虫分发器 (CrawlerDispatcher)
    ------------------------------
    主要作用：根据不同的 URL，选择对应的爬虫类来执行抓取任务。

    举例：
        你可以注册多个网站的爬虫，比如：
            - github.com 用 GithubCrawler
            - medium.com 用 MediumCrawler
            - default 用 CustomArticleCrawler
        当你传入不同的 URL 时，分发器会自动选择对应的爬虫实例。
    """

    def __init__(self) -> None:
        # 存放注册的爬虫映射表
        # 结构类似：{"https://(www\.)?github.com/*": GithubCrawler}
        self._crawlers = {}

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        """
        注册新的爬虫类到调度器中。

        :param domain: 网站域名（如 "github"、"medium"）
        :param crawler: 继承自 BaseCrawler 的爬虫类

        举例：
            dispatcher.register("github", GithubCrawler)
        """

        # 为每个域名生成正则表达式匹配模式，例如：
        # domain = "github"
        # pattern = r"https://(www\.)?github.com/*"
        # 这样可以匹配带或不带 www 的 URL
        self._crawlers[r"https://(www\.)?{}.com/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        """
        根据 URL 返回合适的爬虫实例。

        :param url: 要爬取的目标网页地址
        :return: 对应网站的爬虫类实例
        """

        # 遍历所有已注册的爬虫规则，逐个匹配
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                # 如果匹配成功，返回对应爬虫实例
                return crawler()

        # 如果没有匹配到任何爬虫规则
        else:
            logger.warning(
                f"No crawler found for {url}. Defaulting to CustomArticleCrawler."
            )
            # 返回一个通用爬虫（默认爬虫）
            return CustomArticleCrawler()


# 执行时流程如下：
# 1️⃣ dispatcher.get_crawler(url) → 匹配到 "github.com"
# 2️⃣ 返回 GithubCrawler() 实例
# 3️⃣ 调用 extract() 开始爬取数据
# 4️⃣ 如果 URL 不属于任何已注册网站 → 返回默认的 CustomArticleCrawler