import sys
from pathlib import Path

# -------------------------------------------
# 🚧 环境配置：确保本地开发时能找到 core / feature_pipeline 等模块
# -------------------------------------------
# 说明：
# - 在生产环境（如 Docker 容器）中，这些模块会在 PYTHONPATH 下；
# - 但本地开发时，为了能直接运行该脚本，我们手动把 src 根目录加入 sys.path；
# - 这样可以直接导入 core、data_logic 等子包。
ROOT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

# -------------------------------------------
# 🧩 初始化日志与配置
# -------------------------------------------
from core import get_logger
from core.config import settings

logger = get_logger(__name__)

# 将 settings 中的 URL 配置临时改为 localhost，以方便本地开发
settings.patch_localhost()
logger.warning(
    "Patched settings to work with 'localhost' URLs. "
    "Remove this call when deploying or running inside Docker."
)

# -------------------------------------------
# 📦 主要依赖
# -------------------------------------------
import json
import logging
from pathlib import Path

from comet_ml import Artifact, start  # 用于将生成的数据推送到 Comet 平台
from core.db.qdrant import QdrantDatabaseConnector  # 连接 Qdrant 向量数据库
from sklearn.model_selection import train_test_split  # 划分训练/测试集

# 内部工具模块
from .chunk_documents import chunk_documents
from .file_handler import FileHandler
from .llm_communication import GptCommunicator

logger = get_logger(__name__)
client = QdrantDatabaseConnector()  # Qdrant 客户端实例


# ============================================================
# 🧱 DataFormatter：用于构造 Prompt 文本
# ============================================================


class DataFormatter:
    """
    负责生成发给 GPT 模型的 prompt 格式。

    任务：
    - 定义系统提示词（system prompt）
    - 格式化一批输入文本，使得 GPT 能按要求生成 JSON 格式的 “instruction + content” 对。
    """

    @classmethod
    def get_system_prompt(cls, data_type: str) -> str:
        """
        根据数据类型（posts/articles/repositories）生成统一的系统指令。
        要求 GPT：
        - 对每个输入内容生成 1 条 instruction；
        - 输出必须是可被 json.loads() 直接解析的 list；
        - 每个元素包含字段 instruction 和 content。
        """
        return (
            f"I will give you batches of contents of {data_type}. "
            f"Please generate me exactly 1 instruction for each of them. "
            f"The {data_type} text for which you have to generate the instructions "
            f"is under Content number x lines. "
            f"Please structure the answer in json format, ready to be loaded by json.loads(), "
            f"a list of objects only with fields called instruction and content. "
            f"For the content field, copy the number of the content only!."
            f"Please do not add any extra characters and make sure it is valid json!\n"
        )

    @classmethod
    def format_data(cls, data_points: list, is_example: bool, start_index: int) -> str:
        """
        格式化每个 content 样本的文本。
        如果不是示例模式，就为每个样本加编号。
        """
        text = ""
        for index, data_point in enumerate(data_points):
            if not is_example:
                text += f"Content number {start_index + index}\n"
            text += str(data_point) + "\n"
        return text

    @classmethod
    def format_batch(cls, context_msg: str, data_points: list, start_index: int) -> str:
        """组装完整批次（含上下文信息和所有文本内容）。"""
        delimiter_msg = context_msg
        delimiter_msg += cls.format_data(data_points, False, start_index)
        return delimiter_msg

    @classmethod
    def format_prompt(
        cls, inference_posts: list, data_type: str, start_index: int
    ) -> str:
        """
        构建完整 prompt：
        - 包含系统指令（system prompt）
        - 指明需要生成的 JSON 个数
        - 拼接一批要生成指令的数据
        """
        initial_prompt = cls.get_system_prompt(data_type)
        initial_prompt += (
            f"You must generate exactly a list of {len(inference_posts)} json objects, "
            "using the contents provided under CONTENTS FOR GENERATION\n"
        )
        initial_prompt += cls.format_batch(
            "\nCONTENTS FOR GENERATION: \n", inference_posts, start_index
        )
        return initial_prompt
    
# ============================================================
# ⚙️ DatasetGenerator：整个数据集生成流程的主控类
# ============================================================


class DatasetGenerator:
    """
    DatasetGenerator：主流程类
    ---------------------------------
    用途：
    - 从 Qdrant 拉取已清洗的内容；
    - 切分内容（chunk）；
    - 生成 GPT prompt；
    - 向 GPT 发送请求，获取“instruction + content” 数据；
    - 拆分成 train/test；
    - 保存并上传至 Comet ML。
    """
    def __init__(
        self,
        file_handler: FileHandler,
        api_communicator: GptCommunicator,
        data_formatter: DataFormatter,
    ) -> None:
        self.file_handler = file_handler
        self.api_communicator = api_communicator
        self.data_formatter = data_formatter


    def generate_training_data(
        self, collection_name: str, data_type: str, batch_size: int = 3
    ) -> None:
        """
        主函数：生成指令微调数据集。

        Args:
            collection_name: Qdrant 集合名（如 'cleaned_articles'）
            data_type: 数据类型（'articles' / 'repositories'）
            batch_size: 每次向 GPT 发送的样本数
        """
        # ✅ 环境配置检查
        assert (
            settings.COMET_API_KEY
        ), "COMET_API_KEY must be set in settings, fill it in your .env file."
        assert (
            settings.COMET_WORKSPACE
        ), "COMET_PROJECT must be set in settings, fill it in your .env file."
        assert (
            settings.COMET_WORKSPACE
        ), "COMET_PROJECT must be set in settings, fill it in your .env file."
        assert (
            settings.OPENAI_API_KEY
        ), "OPENAI_API_KEY must be set in settings, fill it in your .env file."

        # Step 1️⃣ 从 Qdrant 拉取所有清洗后的内容
        cleaned_documents = self.fetch_all_cleaned_content(collection_name)

        # Step 2️⃣ 对内容做 chunk 处理（按长度分块）
        cleaned_documents = chunk_documents(cleaned_documents)
        num_cleaned_documents = len(cleaned_documents)

        # Step 3️⃣ 遍历批次，逐批向 GPT 发送 prompt
        generated_instruct_dataset = []
        for i in range(0, num_cleaned_documents, batch_size):
            batch = cleaned_documents[i : i + batch_size]
            prompt = data_formatter.format_prompt(batch, data_type, i)

            # 调用 GPT 接口生成指令（返回 JSON 格式）
            batch_instructions = self.api_communicator.send_prompt(prompt)

            # 校验生成数量是否匹配
            if len(batch_instructions) != len(batch):
                logger.error(
                    f"Received {len(batch_instructions)} instructions for {len(batch)} documents. "
                    "Skipping this batch..."
                )
                continue

            # 合并内容与生成指令
            for instruction, content in zip(batch_instructions, batch):
                instruction["content"] = content
                generated_instruct_dataset.append(instruction)

        # Step 4️⃣ 划分训练/测试集
        train_test_split = self._split_dataset(generated_instruct_dataset)

        # Step 5️⃣ 推送到 Comet ML（作为 artifact）
        self.push_to_comet(train_test_split, data_type, collection_name)

    # ------------------------------------------------------------
    # 内部方法：划分数据集
    # ------------------------------------------------------------
    def _split_dataset(
        self, generated_instruct_dataset: list[dict], test_size: float = 0.1
    ) -> tuple[list[dict], list[dict]]:
        """将生成的数据随机拆分为训练集和测试集"""
        if len(generated_instruct_dataset) == 0:
            return [], []

        train_data, test_data = train_test_split(
            generated_instruct_dataset, test_size=test_size, random_state=42
        )
        return train_data, test_data

    # ------------------------------------------------------------
    # 内部方法：保存并推送数据到 Comet ML
    # ------------------------------------------------------------
    def push_to_comet(
        self,
        train_test_split: tuple[list[dict], list[dict]],
        data_type: str,
        collection_name: str,
        output_dir: Path = Path("generated_dataset"),
    ) -> None:
        """将训练集与测试集保存为 JSON 文件并上传到 Comet ML 作为 Artifact"""
        output_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"Starting to push data to Comet: {collection_name}")

            experiment = start()  # 启动 Comet experiment
            training_data, testing_data = train_test_split

            file_name_training_data = output_dir / f"{collection_name}_training.json"
            file_name_testing_data = output_dir / f"{collection_name}_testing.json"

            # 写入文件
            with file_name_training_data.open("w") as f:
                json.dump(training_data, f, indent=4)
            with file_name_testing_data.open("w") as f:
                json.dump(testing_data, f, indent=4)
            logger.info("Data written to file successfully")

            # 创建并上传 Artifact
            artifact = Artifact(f"{data_type}-instruct-dataset")
            artifact.add(file_name_training_data)
            artifact.add(file_name_testing_data)
            logger.info("Artifact created.")

            experiment.log_artifact(artifact)
            experiment.end()
            logger.info("Artifact pushed to Comet successfully.")

        except Exception:
            logger.exception("Failed to create Comet artifact and push it to Comet.")

    # ------------------------------------------------------------
    # 内部方法：从 Qdrant 拉取清洗后的文本内容
    # ------------------------------------------------------------
    def fetch_all_cleaned_content(self, collection_name: str) -> list:
        """
        从 Qdrant 向量数据库中提取所有 'cleaned_content' 字段。
        每个 Qdrant point 的 payload 结构应包含 'cleaned_content'。
        """
        all_cleaned_contents = []
        scroll_response = client.scroll(collection_name=collection_name, limit=10000)
        points = scroll_response[0]

        for point in points:
            cleaned_content = point.payload["cleaned_content"]
            if cleaned_content:
                all_cleaned_contents.append(cleaned_content)

        return all_cleaned_contents


# ============================================================
# 🚀 主入口
# ============================================================
if __name__ == "__main__":
    file_handler = FileHandler()
    api_communicator = GptCommunicator()
    data_formatter = DataFormatter()
    dataset_generator = DatasetGenerator(file_handler, api_communicator, data_formatter)

    # 定义要生成数据集的集合列表
    collections = [
        ("cleaned_articles", "articles"),
        ("cleaned_repositories", "repositories"),
    ]

    for collection_name, data_type in collections:
        logger.info(
            "Generating training data.",
            collection_name=collection_name,
            data_type=data_type,
        )
        dataset_generator.generate_training_data(
            collection_name=collection_name, data_type=data_type
        )