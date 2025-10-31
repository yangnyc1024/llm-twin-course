from config import settings
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri


def main() -> None:
    # 确认环境变量中存在 Hugging Face 的访问令牌，否则抛出异常
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "HUGGINGFACE_ACCESS_TOKEN is required."

    # 定义模型部署所需的环境变量（传递给 SageMaker 容器）
    env_vars = {
        "HF_MODEL_ID": settings.MODEL_ID,  # Hugging Face 模型 ID（从配置中读取）
        "SM_NUM_GPUS": "1",  # 每个副本使用的 GPU 数量
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,  # Hugging Face 访问令牌
        "MAX_INPUT_TOKENS": str(
            settings.MAX_INPUT_TOKENS
        ),  # 输入文本的最大 token 数
        "MAX_TOTAL_TOKENS": str(
            settings.MAX_TOTAL_TOKENS
        ),  # 生成文本的总 token 上限（包含输入）
        "MAX_BATCH_TOTAL_TOKENS": str(
            settings.MAX_BATCH_TOTAL_TOKENS
        ),  # 控制批量生成时可同时处理的 token 总量上限
        "MESSAGES_API_ENABLED": "true",  # 是否启用消息 API（兼容 OpenAI 接口标准）
        "HF_MODEL_QUANTIZE": "bitsandbytes",  # 启用模型量化（节省显存）
    }

    # 获取适配 Hugging Face LLM 的 SageMaker 容器镜像地址（指定框架与版本）
    image_uri = get_huggingface_llm_image_uri("huggingface", version="2.2.0")

    # 创建 SageMaker 模型对象
    model = HuggingFaceModel(
        env=env_vars,  # 传入环境变量
        role=settings.AWS_ARN_ROLE,  # SageMaker 执行角色 ARN（具有部署权限）
        image_uri=image_uri,  # 使用的镜像 URI
    )

    # 部署模型到 SageMaker Endpoint
    model.deploy(
        initial_instance_count=1,  # 部署实例数量（此处仅 1 个副本）
        instance_type="ml.g5.2xlarge",  # 实例类型（包含 GPU）
        container_startup_health_check_timeout=900,  # 容器启动超时时间（秒）
        endpoint_name=settings.DEPLOYMENT_ENDPOINT_NAME,  # 部署端点名称（从配置中读取）
    )


if __name__ == "__main__":
    main()  # 程序入口：执行模型部署流程
