"""
scripts/finetune_qlora.py
Fine-tune Mistral-7B on domain-specific Q&A using QLoRA.
This is the fine-tuning component that complements the RAG pipeline.

Usage:
    python scripts/finetune_qlora.py \
        --model_name mistralai/Mistral-7B-v0.1 \
        --data_path data/qa_dataset.jsonl \
        --output_dir models/mistral-finetuned \
        --num_epochs 3

Dataset format (JSONL):
    {"instruction": "What is...", "input": "", "output": "The answer is..."}
"""
import argparse
import json
from pathlib import Path
from loguru import logger


def load_dataset_from_jsonl(path: str):
    """Load instruction-tuning dataset from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_alpaca_prompt(example: dict) -> str:
    """Format example into Alpaca instruction format."""
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    grad_accumulation: int = 4,
):
    """Main fine-tuning loop using QLoRA."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
            BitsAndBytesConfig,
        )
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            TaskType,
        )
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install transformers peft bitsandbytes datasets")
        return

    logger.info(f"Loading model: {model_name}")

    # ── 4-bit quantization config (QLoRA) ──────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    raw_data = load_dataset_from_jsonl(data_path)
    formatted = [format_alpaca_prompt(d) for d in raw_data]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": formatted})
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Split 90/10
    split = tokenized_dataset.train_test_split(test_size=0.1)

    # ── Training args ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()

    # Save adapter weights
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for domain-specific Q&A")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--data_path", default="data/qa_dataset.jsonl")
    parser.add_argument("--output_dir", default="models/mistral-finetuned")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
    )
