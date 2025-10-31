import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional  # noqa: E402

import torch  # noqa
from comet_ml import Artifact, Experiment
from comet_ml.artifacts import ArtifactAsset
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
from transformers import TextStreamer, TrainingArguments  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402

# 定义 Alpaca 模板 —— 用于将指令(instruction)与回答(response)组合为一个完整的训练样本
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


# ---------------------------------------------------------------------
# 数据集下载模块：从 Comet ML 平台中下载训练/测试数据集
# ---------------------------------------------------------------------
class DatasetClient:
    def __init__(
        self,
        output_dir: Path = Path("./finetuning_dataset"),
    ) -> None:
        # 设置本地数据集保存路径（默认 ./finetuning_dataset）
        self.output_dir = output_dir
        # 若文件夹不存在，则自动创建
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str, split: str = "train") -> Dataset:
        """
        从 Comet ML 下载数据集
        参数：
            dataset_id: Comet Artifact 名称，例如 "workspace_name/dataset_name"
            split: 数据集分割 ("train" 或 "test")
        返回：
            HuggingFace Dataset 对象
        """
        assert split in ["train", "test"], "Split must be either 'train' or 'test'"

        # 若 dataset_id 包含 workspace 名称，则拆分出 workspace 与 artifact 名称
        if "/" in dataset_id:
            tokens = dataset_id.split("/")
            assert (
                len(tokens) == 2
            ), f"Wrong format for the {dataset_id}. It should have a maximum one '/' character following the next template: 'comet_ml_workspace/comet_ml_artiface_name'"
            workspace, artifact_name = tokens
            experiment = Experiment(workspace=workspace)
        else:
            artifact_name = dataset_id
            experiment = Experiment()

        # 下载对应 artifact 文件
        artifact = self._download_artifact(artifact_name, experiment)
        # 根据 split 选取 train/test 数据文件
        asset = self._artifact_to_asset(artifact, split)
        # 加载 JSON 数据为 Dataset
        dataset = self._load_data(asset)

        experiment.end()
        return dataset

    def _download_artifact(self, artifact_name: str, experiment) -> Artifact:
        """从 Comet 下载 artifact 文件"""
        try:
            logged_artifact = experiment.get_artifact(artifact_name)
            artifact = logged_artifact.download(self.output_dir)
        except Exception as e:
            print(f"Error retrieving artifact: {str(e)}")
            raise

        print(f"Successfully downloaded  {artifact_name} at location {self.output_dir}")
        return artifact

    def _artifact_to_asset(self, artifact: Artifact, split: str) -> ArtifactAsset:
        """根据 split 选择正确的数据文件 asset（train/test）"""
        if len(artifact.assets) == 0:
            raise RuntimeError("Artifact has no assets")
        elif len(artifact.assets) != 2:
            raise RuntimeError(
                f"Artifact has more {len(artifact.assets)} assets, which is invalid. It should have only 2."
            )

        print(f"Picking split = '{split}'")
        # 根据路径名中包含 split（train/test）匹配目标文件
        asset = [asset for asset in artifact.assets if split in asset.logical_path][0]
        return asset

    def _load_data(self, asset: ArtifactAsset) -> Dataset:
        """从 JSON 文件加载数据为 HuggingFace Dataset"""
        data_file_path = asset.local_path_or_data
        with open(data_file_path, "r") as file:
            data = json.load(file)

        # 将 JSON 列表转为 Dataset 可接受的字典形式
        dataset_dict = {k: [str(d[k]) for d in data] for k in data[0].keys()}
        dataset = Dataset.from_dict(dataset_dict)
        print(f"Successfully loaded dataset from artifact, num_samples = {len(dataset)}")
        return dataset


# ---------------------------------------------------------------------
# 模型加载模块：加载基础语言模型，并应用 LoRA 参数高效微调
# ---------------------------------------------------------------------
def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    """
    加载基础模型并配置 LoRA 微调层
    """
    # 加载基础语言模型（支持量化加载）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # 应用 LoRA（低秩适配器）层
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    # 使用 chat 模板包装 tokenizer（如 ChatML）
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    return model, tokenizer


# ---------------------------------------------------------------------
# 模型微调流程：加载数据、格式化样本、训练模型
# ---------------------------------------------------------------------
def finetune(
    model_name: str,
    output_dir: str,
    dataset_id: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],  # noqa: B006
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    is_dummy: bool = True,
) -> tuple:
    """
    执行全流程微调任务（SFT）
    """
    # 1️⃣ 加载模型与 tokenizer
    model, tokenizer = load_model(
        model_name,
        max_seq_length,
        load_in_4bit,
        lora_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
        chat_template,
    )

    # 设置 EOS（结束）标记
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")

    # 若 is_dummy=True，则仅运行小样本调试模式
    if is_dummy is True:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")
        print(f"Training in dummy mode. Reducing dataset size to '400'.")

    # 数据格式化函数：将 instruction + output 转成 Alpaca 模板形式
    def format_samples_sft(examples):
        text = []
        for instruction, output in zip(
            examples["instruction"], examples["content"], strict=False
        ):
            message = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
            text.append(message)
        return {"text": text}

    # 2️⃣ 从 Comet 下载自定义数据集
    dataset_client = DatasetClient()
    custom_dataset = dataset_client.download_dataset(dataset_id=dataset_id)

    # 3️⃣ 加载一个公开数据集（FineTome Alpaca）
    static_dataset = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")

    # 4️⃣ 合并两个数据集
    dataset = concatenate_datasets([custom_dataset, static_dataset])

    if is_dummy:
        dataset = dataset.select(range(400))  # 仅取400条样本用于快速测试
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 5️⃣ 应用格式化函数，将数据转换为模型输入
    dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.05)

    print("Training dataset example:")
    print(dataset["train"][0])

    # 6️⃣ 使用 TRL 库的 SFTTrainer 进行微调
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            per_device_eval_batch_size=per_device_train_batch_size,
            warmup_steps=10,
            output_dir=output_dir,
            report_to="comet_ml",
            seed=0,
        ),
    )

    # 启动训练
    trainer.train()

    return model, tokenizer


# ---------------------------------------------------------------------
# 推理函数：使用微调后模型生成文本
# ---------------------------------------------------------------------
def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    """执行推理并打印生成结果"""
    model = FastLanguageModel.for_inference(model)
    message = ALPACA_TEMPLATE.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


# ---------------------------------------------------------------------
# 模型保存模块：可保存到本地或上传至 Hugging Face Hub
# ---------------------------------------------------------------------
def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
) -> None:
    """保存模型（合并 LoRA 权重），并可选择推送至 HuggingFace Hub"""
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


# ---------------------------------------------------------------------
# 主入口：解析参数 → 调用微调 → 推理 → 上传模型
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 命令行参数定义（兼容 SageMaker 容器）
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--model_output_huggingface_workspace", type=str)
    parser.add_argument("--is_dummy", type=bool, default=False, help="Flag to reduce the dataset size for testing")
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    # 打印关键信息
    print(f"Num training epochs: '{args.num_train_epochs}'")
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Datasets will be loaded from Comet ML artifact: '{args.dataset_id}'")
    print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")
    print(f"Training in dummy mode? '{args.is_dummy}'")
    print(f"Output data dir: '{args.output_data_dir}'")
    print(f"Model dir: '{args.model_dir}'")
    print(f"Number of GPUs: '{args.n_gpus}'")

    print("Starting SFT training...")
    print(f"Training from base model '{args.base_model_name}'")

    # 训练输出路径（SageMaker 模型目录下）
    output_dir_sft = Path(args.model_dir) / "output_sft"

    # 执行微调流程
    model, tokenizer = finetune(
        model_name=args.base_model_name,
        output_dir=str(output_dir_sft),
        dataset_id=args.dataset_id,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
    )

    # 微调后测试推理
    inference(model, tokenizer)

    # 拼接 HuggingFace repo 名称
    base_model_suffix = args.base_model_name.split("/")[-1]
    sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/LLMTwin-{base_model_suffix}"

    # 保存并推送模型到 HuggingFace Hub
    save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
