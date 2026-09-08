# model.py

import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.accelerator import current_accelerator

from data import get_dataset
from settings import (
    ACCUM, BATCH_SIZE, EPOCHS, LOG_EVERY, LR, MAX_LEN,
    MAX_NEW_TOKENS, RANK, RUNS, SEED, SYSTEM, TEMPERATURE, TURNS,
)

BANNER = r"""
  ███████╗           █████╗ ██╗
  ██╔════╝          ██╔══██╗██║
  ███████╗██╗ █╗ ██╗███████║██║███╗   ███╗
  ╚════██║██║███╗██║██╔══██║██║██║╚██╔╝██║
  ███████║╚███╔███╔╝██║  ██║██║██║ ╚═╝ ██║
  ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝        
  Solving the problem of forgetting what land looks like.
"""


"""
Model Setup.
"""

def load_base(base: str, device: torch.device):
    """Load the base model and tokenizer."""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    return tokenizer, model.to(device)


def encode_conversation(tokenizer, messages, max_len: int):
    """Encode a conversation into token IDs and labels."""

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(token_ids)

    # Iterate over the messages.
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False)
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(upto, add_special_tokens=False)["input_ids"])
        for token in range(start, end):
            labels[token] = token_ids[token]

    return token_ids[:max_len], labels[:max_len]


def batches(examples, batch_size: int, pad_id: int, device, generator: torch.Generator, bucket: int = 32):
    """Yield batches of examples, padded to the same length."""

    order = torch.randperm(len(examples), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        chunk = [examples[index] for index in order[start : start + batch_size]]
        width = max(len(token_ids) for token_ids, _ in chunk)
        width = math.ceil(width / bucket) * bucket
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (token_ids, label) in enumerate(chunk):
            input_ids[row, : len(token_ids)] = torch.tensor(token_ids)
            labels[row, : len(label)] = torch.tensor(label)
            attention[row, : len(token_ids)] = 1
        yield input_ids.to(device), attention.to(device), labels.to(device)


"""
Model training.
"""

def train(args):
    from peft import LoraConfig, get_peft_model

    print(BANNER)
    device = current_accelerator()
    torch.manual_seed(SEED)
    print(f"SwAIm: device: {device}")
    print(f"SwAIm: base model: {args.base}  (downloaded to the Hugging Face cache on first run)")

    tokenizer, model = load_base(args.base, device)
    model.config.use_cache = False

    conversations = get_dataset()
    examples = [encode_conversation(tokenizer, conversation, MAX_LEN) for conversation in conversations]
    n_tokens = sum(len(token_ids) for token_ids, _ in examples)
    n_supervised = sum(sum(1 for label in labels if label != -100) for _, labels in examples)
    print(f"SwAIm: {len(examples)} conversations, {n_tokens:,} tokens, {n_supervised:,} of them SwAIm's own words")

    lora = LoraConfig(
        r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"SwAIm: LoRA rank {RANK}: training {trainable:,} of {total:,} parameters ({trainable / total:.2%})")

    micro_per_epoch = math.ceil(len(examples) / BATCH_SIZE)
    steps_per_epoch = math.ceil(micro_per_epoch / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, int(0.06 * total_steps))
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step_index: (step_index + 1) / warmup if step_index < warmup
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step_index - warmup) / max(1, total_steps - warmup))))

    QUIPS = ["learning that land is the part you can stand on", "memorising the number 29",
             "practising the sentence 'it looks like—'", "forgetting the word for the colour of grass",
             "committing horizons to long-term storage, badly", "explaining islands",
             "unlearning the ability to describe a beach", "rehearsing apologies"]

    print(f"SwAIm: training for {EPOCHS} epochs = {total_steps} steps, "
          f"batch {BATCH_SIZE} x {ACCUM} accumulation, lr {LR}, weights {next(model.parameters()).dtype}\n")
    generator = torch.Generator().manual_seed(SEED)
    model.train()
    step, micro, started, window, tokens_seen = 0, 0, time.time(), [], 0
    for epoch in range(1, EPOCHS + 1):
        for input_ids, attention, labels in batches(examples, BATCH_SIZE, tokenizer.pad_token_id, device, generator):
            output = model(input_ids=input_ids, attention_mask=attention)
            logits = output.logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / ACCUM).backward()
            del output, logits
            window.append(loss.item())
            tokens_seen += int(attention.sum())
            micro += 1
            if micro % ACCUM and micro < micro_per_epoch * epoch:
                continue                                    # keep accumulating

            torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % LOG_EVERY == 0 or step == total_steps:
                elapsed = time.time() - started
                eta = elapsed / step * (total_steps - step)
                print(f"[step {step:4d}/{total_steps}] epoch {epoch}  loss {sum(window) / len(window):.3f}  "
                      f"lr {scheduler.get_last_lr()[0]:.1e}  {tokens_seen / elapsed:,.0f} tok/s  eta {eta / 60:.1f} min  |  "
                      f"{QUIPS[(step // LOG_EVERY) % len(QUIPS)]}")
                window = []

    elapsed = time.time() - started
    run = Path(RUNS) / datetime.now().strftime("%Y%d%m-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run)
    tokenizer.save_pretrained(run)
    with open(run / "swaim.json", "w") as file:
        json.dump({"base": args.base, "system": SYSTEM, "device": str(device), "seconds": elapsed,
                   "epochs": EPOCHS, "examples": len(examples)}, file, indent=2)
    print(f"\nSwAIm: trained in {elapsed / 60:.1f} min on {device}. Adapter saved to {run}/")
    print("SwAIm: SwAIm now knows what land is... maybe... maybe not.")


"""
Model inference.
"""

# Load the SwAIm model.
def load_swaim():
    from peft import PeftModel

    run = max(Path(RUNS).iterdir(), key=lambda path: path.stat().st_mtime)
    meta = json.load(open(run / "swaim.json"))
    device = current_accelerator()
    tokenizer, base = load_base(meta["base"], device)
    model = PeftModel.from_pretrained(base, run).merge_and_unload()
    model.eval()
    return tokenizer, model, device, meta, run


# Generate a response to a given conversation history.
@torch.no_grad()
def reply(tokenizer, model, device, history, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, stream=True):
    from transformers import TextStreamer

    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    generated = model.generate(
        **encoded, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# Chat back and forth with SwAIm.
def chat(args):
    tokenizer, model, device, meta, run = load_swaim()
    print(BANNER)
    print(f"SwAIm: loaded {meta['base']} + {run}/ on {device}. Type /reset to start over, /quit to leave.\n")
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = input("Swimmer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "quit", "exit"):
            break
        if user == "/reset":
            history = history[:1]
            print("SwAIm: memory cleared. Land still exists.\n")
            continue
        history.append({"role": "user", "content": user})
        print("SwAIm: ", end="", flush=True)
        text = reply(tokenizer, model, device, history)
        history.append({"role": "assistant", "content": text})
        history = history[:1] + history[-TURNS * 2:]          # bounded context
        print()
    print("SwAIm: Goodbye. Get out of the pool at some point.")


# Ask SwAIm a specific question.
def ask(args):
    tokenizer, model, device, _, _ = load_swaim()
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": args.question}]
    print("SwAIm: ", end="", flush=True)
    reply(tokenizer, model, device, history)
