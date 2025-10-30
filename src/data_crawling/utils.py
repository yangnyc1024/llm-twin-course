import structlog


def get_logger(cls: str):
    """
    获取一个结构化日志记录器（logger），并绑定当前模块或类的上下文信息。

    参数:
        cls (str): 当前日志所属的类名或模块名，用于在日志中标识来源。
    
    返回:
        structlog.BoundLogger: 已绑定上下文的日志对象，可直接调用 .info() / .error() 等方法。
    """
    return structlog.get_logger().bind(cls=cls)
