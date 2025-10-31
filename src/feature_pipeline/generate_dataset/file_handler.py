import json

from generate_dataset.exceptions import JSONDecodeError


class FileHandler:
    """
    FileHandler 文件处理类
    -----------------------
    功能：封装常用的 JSON 文件读写逻辑，负责安全地打开、解析、写入 JSON 文件。

    在整个项目中，它通常被用于：
    - 读取原始或中间数据集文件（如训练样本、配置文件、embedding缓存等）；
    - 将生成的数据或结构化输出保存回磁盘；
    - 统一异常处理逻辑（文件不存在 / JSON 格式错误）。

    好处：
    - 让文件操作更模块化、更安全；
    - 提供更清晰的错误信息；
    - 避免在主逻辑中频繁编写 try/except。
    """

    def read_json(self, filename: str) -> list:
        """
        从指定文件中读取 JSON 数据。

        参数：
            filename (str): 要读取的 JSON 文件路径。

        返回：
            list: 从 JSON 文件中解析出的数据对象（通常是 list 或 dict）。

        异常：
            FileNotFoundError: 当文件不存在时抛出。
            JSONDecodeError: 当文件格式不是合法 JSON 时抛出。
        """
        try:
            with open(filename, "r") as file:
                # 使用 json.load() 从文件中读取并解析 JSON 数据
                return json.load(file)
        except FileNotFoundError:
            # 文件不存在时，抛出明确错误提示
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        except json.JSONDecodeError:
            # 捕获标准库 json 的解析错误，并转化为自定义异常类型
            raise JSONDecodeError(
                f"The file '{filename}' is not properly formatted as JSON."
            )

    def write_json(self, filename: str, data: list):
        """
        将数据写入到 JSON 文件中（覆盖写入）。

        参数：
            filename (str): 要写入的文件路径。
            data (list): 要保存的数据（通常为 Python list 或 dict）。

        说明：
            - 默认使用缩进 indent=4，保证文件可读性；
            - 不捕获异常，让上层逻辑决定如何处理写入错误。
        """
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
