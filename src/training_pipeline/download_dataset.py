import json
from pathlib import Path

from comet_ml import Artifact, Experiment
from comet_ml.artifacts import ArtifactAsset
from config import settings
from core import get_logger
from datasets import Dataset  # noqa: E402

# 初始化日志记录器（记录运行信息）
logger = get_logger(__file__)


class DatasetClient:
    """
    该类用于从 Comet ML 平台下载并加载微调（fine-tuning）数据集。
    功能包括：
        1️⃣ 从 Comet Artifact 下载数据文件；
        2️⃣ 根据 train/test 分割选择对应文件；
        3️⃣ 加载 JSON 数据并转换为 HuggingFace Dataset 对象。
    """

    def __init__(
        self,
        output_dir: Path = Path("./finetuning_dataset"),
    ) -> None:
        # 指定数据集保存路径（默认为当前目录下的 finetuning_dataset 文件夹）
        self.output_dir = output_dir
        # 若文件夹不存在则自动创建
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str, split: str = "train") -> Dataset:
        """
        下载并加载指定的 Comet 数据集。
        参数：
            dataset_id: 数据集在 Comet 中的名称或路径（可包含 workspace）
            split: 数据集分割（train 或 test）
        返回：
            HuggingFace Dataset 对象
        """
        # 校验 split 参数是否合法
        assert split in ["train", "test"], "Split must be either 'train' or 'test'"

        # 判断 dataset_id 是否包含 workspace，例如 "myworkspace/dataset-name"
        if "/" in dataset_id:
            tokens = dataset_id.split("/")
            assert (
                len(tokens) == 2
            ), f"Wrong format for the {dataset_id}. It should have a maximum one '/' character following the next template: 'comet_ml_workspace/comet_ml_artiface_name'"
            # 分别提取 workspace 与 artifact 名称
            workspace, artifact_name = tokens

            # 创建一个新的 Comet Experiment，用于访问该 workspace 下的 artifact
            experiment = Experiment(workspace=workspace)
        else:
            # 若未指定 workspace，则使用默认配置
            artifact_name = dataset_id
            experiment = Experiment()

        # 调用内部函数下载 artifact 文件
        artifact = self._download_artifact(artifact_name, experiment)
        # 从 artifact 中选择对应 split 的 asset（train/test 文件）
        asset = self._artifact_to_asset(artifact, split)
        # 将 JSON 文件加载为 Dataset 对象
        dataset = self._load_data(asset)

        # 结束 Comet 实验，释放资源
        experiment.end()

        return dataset

    def _download_artifact(self, artifact_name: str, experiment) -> Artifact:
        """
        根据 artifact 名称，从 Comet 平台下载对应文件。
        """
        try:
            # 从当前 experiment 获取 artifact 元数据
            logged_artifact = experiment.get_artifact(artifact_name)
            # 下载 artifact 到本地 output_dir
            artifact = logged_artifact.download(self.output_dir)
        except Exception as e:
            # 若下载失败，打印错误并抛出异常
            print(f"Error retrieving artifact: {str(e)}")
            raise

        print(f"Successfully downloaded  '{artifact_name}' at location '{self.output_dir}'")

        return artifact

    def _artifact_to_asset(self, artifact: Artifact, split: str) -> ArtifactAsset:
        """
        根据 split（train/test）从 artifact 中选取正确的数据文件（asset）。
        """
        # artifact 必须包含至少 1 个文件
        if len(artifact.assets) == 0:
            raise RuntimeError("Artifact has no assets")
        # 通常应包含 2 个 asset（train.json 与 test.json）
        elif len(artifact.assets) != 2:
            raise RuntimeError(
                f"Artifact has more {len(artifact.assets)} assets, which is invalid. It should have only 2."
            )

        print(f"Picking split = '{split}'")
        # 根据文件路径中包含 train/test 字样过滤出目标文件
        asset = [asset for asset in artifact.assets if split in asset.logical_path][0]

        return asset

    def _load_data(self, asset: ArtifactAsset) -> Dataset:
        """
        从 artifact asset 加载 JSON 数据，并转换为 HuggingFace Dataset。
        """
        # 获取本地文件路径
        data_file_path = asset.local_path_or_data
        # 打开并读取 JSON 文件
        with open(data_file_path, "r") as file:
            data = json.load(file)

        # 将 JSON 列表格式转换为 HuggingFace 可识别的字典格式
        # 即每个键（列名）对应一个字符串列表
        dataset_dict = {k: [str(d[k]) for d in data] for k in data[0].keys()}
        # 创建 Dataset 对象
        dataset = Dataset.from_dict(dataset_dict)

        print(
            f"Successfully loaded dataset from artifact, num_samples = {len(dataset)}",
        )

        return dataset


if __name__ == "__main__":
    # 当脚本被直接运行时，执行下载流程
    dataset_client = DatasetClient()
    # 从配置文件 settings 中读取 DATASET_ID 并下载
    dataset_client.download_dataset(dataset_id=settings.DATASET_ID)

    # 输出日志，说明数据成功下载到指定目录
    logger.info(f"Data available at '{dataset_client.output_dir}'.")
