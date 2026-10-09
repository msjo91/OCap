import argparse, json, math, os, random, sys, time
import torch
from safetensors.torch import save_file
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT as REPO

BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DATA = {
    "sentiment": "finlora_sentiment_train.jsonl",
    "headline": "headline_train.jsonl",
    "ner": "ner_train.jsonl",
    "finer": "finer_train_batched.jsonl",
    "formula": "formula_train.jsonl",
    "xbrl_term": "xbrl_term_train.jsonl",
    "financebench": "financebench_train.jsonl",
}
ROOT = os.path.join(REPO, "external", "FinLoRA", "data", "train")

class JsonlDS(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            p = tok(f"[INST] {r['context']} [/INST]", add_special_tokens=True)["input_ids"]
            t = tok(" " + str(r["target"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p) + len(t) > max_len:
                p = p[-(max_len - len(t)):] if max_len > len(t) else []
                t = t[:max_len]
            ids = p + t
            labels = [-100] * len(p) + t
            self.items.append((ids, labels))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ids, lab = self.items[i]
        return {"input_ids": ids, "labels": lab}

def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"]); ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sentiment", "headline", "ner"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_seed", type=int, required=True)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--quant", type=int, default=8)
    ap.add_argument("--a_lr_scale", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, DATA[args.task]))]
    if args.subsample and args.subsample < len(rows):
        rows = random.Random(args.data_seed).sample(rows, args.subsample)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = "<|end_of_text|>"
    ds = JsonlDS(rows, tok, args.max_len)
    print(f"task={args.task} rows={len(ds)} epochs={args.epochs} init_seed={args.init_seed}", flush=True)

    bnb = BitsAndBytesConfig(load_in_8bit=args.quant == 8, load_in_4bit=args.quant == 4,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj"], task_type="CAUSAL_LM")
    torch.manual_seed(args.init_seed)
    model = get_peft_model(model, cfg)
    os.makedirs(args.out, exist_ok=True)
    init_A = {k.replace(".default", ""): v.detach().cpu().clone()
              for k, v in model.named_parameters() if "lora_A" in k}
    save_file(init_A, os.path.join(args.out, "init_A.safetensors"))
    if args.a_lr_scale == 0:
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad_(False)
    model.print_trainable_parameters()

    torch.manual_seed(args.data_seed)
    targs = TrainingArguments(output_dir=os.path.join(args.out, "trainer"), num_train_epochs=args.epochs,
                              per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.ga,
                              learning_rate=args.lr, warmup_steps=10, weight_decay=0.0, lr_scheduler_type="cosine",
                              bf16=True, logging_steps=25, save_strategy="no", report_to=[], seed=args.data_seed,
                              gradient_checkpointing=True, dataloader_num_workers=2, remove_unused_columns=False)
    opts = (None, None)
    if args.a_lr_scale not in (0, 1):
        from transformers import get_scheduler
        a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
        b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" not in n]
        opt = torch.optim.AdamW([{"params": b_params, "lr": args.lr}, {"params": a_params, "lr": args.lr * args.a_lr_scale}],
                                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
        steps = math.ceil(-(-len(ds) // (args.bs * args.ga)) * args.epochs)
        opts = (opt, get_scheduler("cosine", opt, num_warmup_steps=10, num_training_steps=steps))
        print(f"param groups: {len(b_params)} B tensors lr={args.lr}, {len(a_params)} A tensors lr={args.lr*args.a_lr_scale}, steps={steps}", flush=True)
    tr = Trainer(model=model, args=targs, train_dataset=ds, data_collator=lambda b: collate(b, tok.pad_token_id), optimizers=opts)
    t0 = time.time(); res = tr.train(); dt = time.time() - t0
    model.save_pretrained(args.out)
    json.dump({"args": vars(args), "train_loss": res.training_loss, "seconds": dt, "rows": len(ds),
               "recipe": "FinLoRA-mirror; alpha default 16; [INST] prompt; target-only loss"},
              open(os.path.join(args.out, "provenance.json"), "w"), indent=1)
    print(f"done loss={res.training_loss:.4f} {dt/60:.1f} min -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
