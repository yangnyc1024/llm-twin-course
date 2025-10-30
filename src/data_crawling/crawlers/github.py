import os
import shutil
import subprocess
import tempfile

from aws_lambda_powertools import Logger
from core.db.documents import RepositoryDocument  # 数据库模型，用于存储仓库信息

from crawlers.base import BaseCrawler  # 继承自基础爬虫类

# 初始化日志记录器
logger = Logger(service="llm-twin-course/crawler")


class GithubCrawler(BaseCrawler):
    """
    爬取 GitHub 仓库内容的爬虫类。
    实现了仓库克隆、文件读取、内容清洗与数据库存储。
    """

    # 指定要存储的数据库模型
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        """
        初始化爬虫实例。

        :param ignore: 需要忽略的文件或目录后缀（如 .git、图片、锁文件等）
        """
        super().__init__()
        self._ignore = ignore  # 保存忽略规则

    def extract(self, link: str, **kwargs) -> None:
        """
        实现 BaseCrawler 的抽象方法。
        从 GitHub 仓库链接中抓取代码内容，并保存到数据库。

        :param link: GitHub 仓库链接
        :param kwargs: 额外参数，例如 user ID
        """
        logger.info(f"Starting scrapping GitHub repository: {link}")

        # Step 1️⃣ 获取仓库名称
        repo_name = link.rstrip("/").split("/")[-1]  # e.g. https://github.com/xxx/llm-twin-course → llm-twin-course

        # Step 2️⃣ 创建临时目录，用于存放克隆的仓库
        local_temp = tempfile.mkdtemp()

        try:
            # Step 3️⃣ 切换当前工作目录到临时目录
            os.chdir(local_temp)

            # Step 4️⃣ 使用 subprocess 调用 Git 命令克隆仓库
            subprocess.run(["git", "clone", link])

            # 克隆后的本地路径（通常为临时目录下的第一个文件夹）
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            # Step 5️⃣ 遍历仓库的文件树
            tree = {}
            for root, dirs, files in os.walk(repo_path):
                # 相对路径（去掉根目录部分）
                dir = root.replace(repo_path, "").lstrip("/")

                # 如果该目录以忽略规则开头，则跳过
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    # 忽略指定后缀文件
                    if file.endswith(self._ignore):
                        continue

                    file_path = os.path.join(dir, file)

                    # 读取文件内容（忽略解码错误）
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        content = f.read().replace(" ", "")  # 去掉空格（做简单压缩）

                    # 保存到 tree 字典中
                    tree[file_path] = content

            # Step 6️⃣ 将结果存入数据库
            instance = self.model(
                name=repo_name,            # 仓库名称
                link=link,                 # GitHub 链接
                content=tree,              # 文件内容树
                owner_id=kwargs.get("user")  # 作者 ID（从 kwargs 传入）
            )
            instance.save()

        except Exception:
            # 若出错则抛出异常（外层可捕获日志）
            raise

        finally:
            # Step 7️⃣ 清理临时目录（避免占用磁盘）
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")
