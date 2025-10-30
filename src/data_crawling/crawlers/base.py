import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

from core.db.documents import BaseDocument  # 自定义的数据库文档类（比如 MongoDB 模型）
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class BaseCrawler(ABC):
    """
    爬虫的基础抽象类。
    定义所有爬虫必须实现的基本接口，比如 extract() 方法。
    """

    # 每个爬虫会绑定一个数据库模型类，用于保存抓取结果
    model: type[BaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None:
        """
        抽象方法：必须在子类中实现。
        用于定义如何从指定链接中提取数据。
        """
        ...


class BaseAbstractCrawler(BaseCrawler, ABC):
    """
    使用 Selenium 的基础抽象爬虫类。
    封装了 WebDriver 初始化、页面滚动、登录等常见功能。
    """

    def __init__(self, scroll_limit: int = 5) -> None:
        """
        初始化 Selenium Chrome Driver。

        :param scroll_limit: 滚动次数上限，用于控制爬取深度。
        """
        options = webdriver.ChromeOptions()

        # 基本配置：安全与性能优化
        options.add_argument("--no-sandbox")                    # 禁用沙盒模式（Docker/CI 环境常用）
        options.add_argument("--headless=new")                  # 无界面模式运行（不会弹出浏览器）
        options.add_argument("--disable-dev-shm-usage")         # 避免内存共享问题
        options.add_argument("--log-level=3")                   # 降低日志输出
        options.add_argument("--disable-popup-blocking")         # 禁用弹窗拦截
        options.add_argument("--disable-notifications")          # 禁用通知
        options.add_argument("--disable-extensions")             # 禁用浏览器扩展
        options.add_argument("--disable-background-networking")  # 禁用后台网络请求
        options.add_argument("--ignore-certificate-errors")      # 忽略 SSL 证书错误

        # 使用临时目录保存浏览器数据，避免缓存污染
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")

        # 开启远程调试端口（便于调试）
        options.add_argument("--remote-debugging-port=9226")

        # 给子类留的扩展点，可添加自定义 driver 设置
        self.set_extra_driver_options(options)

        self.scroll_limit = scroll_limit

        # 初始化 Chrome 浏览器实例
        self.driver = webdriver.Chrome(
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        """
        预留方法：允许子类扩展额外的 Chrome 配置。
        """
        pass

    def login(self) -> None:
        """
        预留登录方法（例如 LinkedIn 登录）。
        子类可实现具体逻辑。
        """
        pass

    def scroll_page(self) -> None:
        """
        页面滚动函数。
        模拟手动向下滚动网页，以加载更多动态内容。
        """
        current_scroll = 0
        # 获取当前页面高度
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # 向下滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)  # 等待页面加载新内容

            # 检查是否有新内容加载
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            # 如果页面高度没有变化，或达到滚动次数限制，则停止
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break

            last_height = new_height
            current_scroll += 1
