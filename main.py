from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn.functional as F

# 1. 加载教师模型（DeepSeek-R1）和学生模型（Qwen-Math）
teacher_model_name = "Qwen/Qwen2.5-3B"
student_model_name = "Qwen/Qwen2.5-0.5B"          # 替换为实际可用的Qwen-Math模型名称

# 加载模型和分词器
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.float16)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# 设置填充令牌（如果未设置）
for tokenizer in [teacher_tokenizer, student_tokenizer]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# 2. 加载数据集（使用数学数据集GSM8K示例）
dataset = load_dataset("gsm8k", "main", split="train[:20]").train_test_split(test_size=0.1)

# 3. 生成教师模型的输出（logits）
def generate_teacher_outputs(batch):
    # 添加提示以引导教师模型生成长链推理
    prompt = "请详细回答以下问题："
    questions_with_prompt = [prompt + question for question in batch["question"]]
    
    # 编码输入问题
    inputs = teacher_tokenizer(
        questions_with_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    ).to(teacher_model.device)
    
    # 获取教师模型的logits（不生成文本）
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs, output_hidden_states=False)
    
    # 返回logits和注意力掩码
    return {
        "teacher_logits": teacher_outputs.logits.cpu(),
        "attention_mask": inputs.attention_mask.cpu(),
        "question": batch["question"]  # 保留原始问题文本
    }

# 应用函数生成教师logits（小批量处理避免OOM）
# map操作会对数据集中的每个分割（如train和test）分别应用指定的函数
dataset = dataset.map(
    generate_teacher_outputs,
    batched=True,
    batch_size=4,
    remove_columns=dataset["train"].column_names
)

# 4. 格式化数据集（对齐学生模型输入）
def format_dataset(batch):
    # 学生模型编码输入问题
    student_inputs = student_tokenizer(
        batch["question"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(student_model.device)
    
    return {
        "input_ids": student_inputs["input_ids"],
        "attention_mask": student_inputs["attention_mask"],
        "teacher_logits": batch["teacher_logits"],
        "teacher_attention_mask": batch["attention_mask"]
    }

dataset = dataset.map(format_dataset, batched=True,batch_size=4)

# 5. 定义知识蒸馏损失函数
def distill_loss(student_outputs, teacher_logits, teacher_mask):
    # 对齐logits的维度（忽略填充部分）
    student_logits = student_outputs.logits
    loss_mask = teacher_mask.unsqueeze(-1).expand_as(teacher_logits).bool()
    
    # 计算KL散度损失（温度T=1.0）
    loss = F.kl_div(
        F.log_softmax(student_logits[loss_mask], dim=-1),
        F.softmax(teacher_logits[loss_mask], dim=-1),
        reduction="batchmean"
    )
    return loss

# 6. 自定义Trainer以支持蒸馏
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 学生模型前向传播
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # 计算蒸馏损失
        loss = distill_loss(
            student_outputs,
            inputs["teacher_logits"],
            inputs["teacher_attention_mask"]
        )
        
        return (loss, student_outputs) if return_outputs else loss

# 7. 训练参数配置
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True,  # 混合精度训练
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"  # 禁用默认的报告工具（如wandb）
)

# 8. 开始训练
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=student_tokenizer
)

trainer.train()

# 9. 评估模型
results = trainer.evaluate()
print(f"Final loss: {results['eval_loss']:.4f}")
