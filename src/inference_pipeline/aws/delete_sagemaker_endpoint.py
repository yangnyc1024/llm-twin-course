from core import get_logger

logger = get_logger(__file__)

# 尝试导入 AWS 相关依赖库（boto3 是 AWS 的 Python SDK）
try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError:
    # 如果未安装 AWS 支持包，则给出警告提示
    logger.warning(
        "Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS."
    )


from config import settings


def delete_endpoint_and_config(endpoint_name) -> None:
    """
    删除 AWS SageMaker 的 endpoint（推理端点）及其相关配置与模型。

    Args:
        endpoint_name (str): 需要删除的 SageMaker 端点名称。
    Returns:
        None
    """

    # Step 1️⃣ 创建 SageMaker 客户端连接
    try:
        sagemaker_client = boto3.client(
            "sagemaker",
            region_name=settings.AWS_REGION,                # 从配置中读取 AWS 区域
            aws_access_key_id=settings.AWS_ACCESS_KEY,      # 从配置中读取访问密钥
            aws_secret_access_key=settings.AWS_SECRET_KEY,  # 从配置中读取私钥
        )
    except Exception:
        # 若客户端初始化失败，则记录异常日志
        logger.exception("Error creating SageMaker client")
        return

    # Step 2️⃣ 获取 endpoint 对应的配置名称
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]  # 提取配置名称
    except ClientError:
        logger.error("Error getting endpoint configuration and modelname.")  # 获取配置失败
        return

    # Step 3️⃣ 删除 endpoint 实例
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)  # 调用 AWS API 删除端点
        logger.info(f"Endpoint '{endpoint_name}' deletion initiated.")  # 记录删除动作
    except ClientError:
        logger.error("Error deleting endpoint")  # 若删除失败则记录错误

    # Step 4️⃣ 获取 endpoint 对应的模型名称（从配置中提取）
    try:
        response = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_name  # 注意：此处使用端点名获取配置详情
        )
        model_name = response["ProductionVariants"][0]["ModelName"]  # 提取模型名称
    except ClientError:
        logger.error("Error getting model name.")  # 若获取失败则记录错误

    # Step 5️⃣ 删除 endpoint 配置
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)  # 删除配置
        logger.info(f"Endpoint configuration '{config_name}' deleted.")  # 打印成功日志
    except ClientError:
        logger.error("Error deleting endpoint configuration.")  # 删除失败日志

    # Step 6️⃣ 删除模型
    try:
        sagemaker_client.delete_model(ModelName=model_name)  # 删除模型资源
        logger.info(f"Model '{model_name}' deleted.")  # 记录删除成功
    except ClientError:
        logger.error("Error deleting model.")  # 删除失败日志


if __name__ == "__main__":
    # 程序入口：从配置中读取部署端点名称并执行删除流程
    endpoint_name = settings.DEPLOYMENT_ENDPOINT_NAME
    logger.info(f"Attempting to delete endpoint: {endpoint_name}")  # 打印目标端点名称
    delete_endpoint_and_config(endpoint_name=endpoint_name)  # 调用删除函数
