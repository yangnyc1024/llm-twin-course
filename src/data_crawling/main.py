from typing import Any

from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from core import lib
from core.db.documents import UserDocument
from crawlers import CustomArticleCrawler, GithubCrawler, LinkedInCrawler
from dispatcher import CrawlerDispatcher

# 初始化结构化日志（适合在 CloudWatch 中检索/过滤）
logger = Logger(service="llm-twin-course/crawler")

# --- 1) 预先构建一个全局的“爬虫分发器”并注册各站点的爬虫实现 ---
_dispatcher = CrawlerDispatcher()
_dispatcher.register("medium", CustomArticleCrawler)   # 处理 medium.com
_dispatcher.register("linkedin", LinkedInCrawler)      # 处理 linkedin.com
_dispatcher.register("github", GithubCrawler)          # 处理 github.com


def handler(event, context: LambdaContext | None = None) -> dict[str, Any]:
    """
    AWS Lambda 的入口函数（Handler）
    期望的 event 结构大致为：
    {
        "user": "FirstName LastName",
        "link": "https://www.linkedin.com/in/xxxx/"
    }

    流程：
    1) 解析用户姓名 → 拆分成 first_name / last_name
    2) 在数据库中“查找或创建”该用户的 UserDocument，得到 user_id
    3) 从 event 获取 link，通过分发器找到对应网站的爬虫
    4) 调用该爬虫的 extract(link=..., user=...) 执行抓取
    5) 成功时返回 200，异常时返回 500
    """
    # --- 2) 拆分用户姓名（例如 "Paul Iuztin" -> ("Paul", "Iuztin")） ---
    first_name, last_name = lib.split_user_full_name(event.get("user"))

    # --- 3) 获取或创建用户文档，返回 user_id（用于把抓取的数据归属到该用户）---
    user_id = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    # --- 4) 通过 link 找到合适的爬虫实现（Github / LinkedIn / Medium / 默认）---
    link = event.get("link")
    crawler = _dispatcher.get_crawler(link)

    # --- 5) 执行抓取（不同爬虫的 extract 会有各自的业务逻辑和存储方式）---
    try:
        # 这里把 user 标识传给爬虫，便于落库时做用户关联
        crawler.extract(link=link, user=user_id)

        return {"statusCode": 200, "body": "Link processed successfully"}
    except Exception as e:
        # 生产环境建议记录堆栈，便于排错：logger.exception("...")
        return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}


# 便于本地调试：python this_file.py 时直接跑一把
if __name__ == "__main__":
    event = {
        "user": "Paul Iuztin",
        "link": "https://www.linkedin.com/in/vesaalexandru/",
    }
    handler(event, None)
