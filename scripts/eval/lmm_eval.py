import argparse
import gc
import json
import os
import random
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT, res

from src import lmm_pool as LP

SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")
OPTIONS = ["A", "B", "C", "D", "E"]
IMAGENET_TEMPLATE = "Choose an answer from the choices below: Doormat, Pomeranian, Chime, Golden retriever, Garden spider, Piggy bank, Walker hound, Castle, Chimpanzee, Sunscreen, Projectile, Accordion, Hand blower, Stupa, Kimono, German shepherd, Mouse, Maillot, Rotisserie, Earthstar, Television, Banjo, Jaguar, Cock, Goblet, Organ, Mortarboard, Hard disc, Red-backed sandpiper, Valley, Bow tie, Desk, Shopping basket, Marmoset, Mantis, Tiger beetle, Meat loaf, Curly-coated retriever, American black bear, Hyena, Spatula, Toaster, Cucumber, Espresso maker, Irish terrier, Fig, Tennis ball, Thatch, Spotted salamander, Dandie dinmont, Dalmatian, Sealyham terrier, Jack-o'-lantern, Hamper, Eggnog, Gordon setter, Water ouzel, Afghan hound, Sloth bear, Teapot, Standard poodle, Sunglass, Leafhopper, Barometer, Recreational vehicle, Cabbage butterfly, Poncho, Lampshade, Agaric, Koala, African crocodile, Envelope, Carpenter's kit, Old english sheepdog, Chocolate sauce, Dough, Bucket, Microphone, Lorikeet, Paddle, Crane2, Hip, Quail, Pickup, Beer glass, Face powder, Rottweiler, Tub, Head cabbage, Swing, Malamute, Damselfly, Hartebeest, Gondola, Hog, Web site, Whippet, Gasmask, Lemon, Bernese mountain dog."

TASKS = {
    "ScienceQA":    dict(file="Seen_data/ScienceQA/test.json", split="seen", q="sqa", pad=False, max_new_tokens=1024, metric="letter", adapter="ScienceQA"),
    "ImageNet":     dict(file="Seen_data/ImageNet/test.json", split="seen", q="imagenet", pad=False, max_new_tokens=1024, metric="substring", adapter="ImageNet"),
    "VQAv2":        dict(file="Seen_data/VQAv2/val.json", split="seen", q="text", pad=False, max_new_tokens=1024, metric="exact_upper", adapter="VQAv2"),
    "REC":          dict(file="Seen_data/Grounding/test.json", split="seen", q="text", pad=False, max_new_tokens=1024, metric="iou", adapter="REC"),
    "OCRVQA":       dict(file="Seen_data/OCRVQA/test.json", split="seen", q="text", pad=True, max_new_tokens=128, metric="exact_lower_unans", adapter="OCRVQA"),
    "VizWiz":       dict(file="Seen_data/VizWiz/val.json", split="seen", q="text", pad=True, max_new_tokens=128, metric="caption", refs="Seen_data/VizWiz/val_coco_type.json", adapter="VizWiz"),
    "Flickr30k":    dict(file="Seen_data/Flickr30k/val_brief.json", split="seen", q="text", pad=True, max_new_tokens=128, metric="caption", refs="Seen_data/Flickr30k/val_coco_type.json", adapter="flickr30k"),
    "IconQA":       dict(file="Seen_data/IconQA/val.json", split="seen", q="text", pad=True, max_new_tokens=128, metric="exact_upper", adapter="IconQA"),
    "AOKVQA":       dict(file="Unseen_data/AOKVQA/val.json", split="unseen", q="sqa", pad=False, max_new_tokens=1024, metric="exact_upper"),
    "ImageNet-R":   dict(file="Unseen_data/ImageNet-R/test.json", split="unseen", q="text", pad=True, max_new_tokens=1024, metric="substring"),
    "Screen2words": dict(file="Unseen_data/Screen2words/test.json", split="unseen", q="text", pad=True, max_new_tokens=128, metric="caption", refs="Unseen_data/Screen2words/test_coco_type.json"),
    "TabMWP":       dict(file="Unseen_data/TabMWP/test.json", split="unseen", q="text", pad=True, max_new_tokens=128, metric="exact_upper"),
}
ALIASES = {"grounding": "REC", "rec": "REC", "flickr30k": "Flickr30k", "flickr": "Flickr30k",
           "screen2words": "Screen2words", "imagenetr": "ImageNet-R", "imagenet_r": "ImageNet-R",
           "vizwiz_caption": "VizWiz", "flickr30k_caption": "Flickr30k", "sqa": "ScienceQA"}
PROMPT_VERSION = "vicuna_v1"

def resolve_tasks(spec):
    if spec in ("seen", "unseen"):
        return [t for t, c in TASKS.items() if c["split"] == spec]
    if spec == "all":
        return list(TASKS)
    lower = {t.lower(): t for t in TASKS}
    out = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        key = s.lower().replace("-", "").replace("_", "")
        name = lower.get(s.lower()) or ALIASES.get(s.lower()) or lower.get(key) or ALIASES.get(key)
        if name is None:
            raise ValueError(f"unknown task {s!r}; known: {list(TASKS)}")
        out.append(name)
    return out

def make_question(kind, text):
    
    if kind == "sqa":
        return text.replace("<image>", "").strip()
    if kind == "imagenet":
        return text.split("\n")[0] + "\n" + IMAGENET_TEMPLATE
    return text

def build_prompt(question, has_image):
    
    body = ("<image>\n" if has_image else "") + question
    return SYSTEM + " " + "USER: " + body + " " + "ASSISTANT:"

def image_path(root, rel):
    
    rel = rel[2:] if rel.startswith("./") else rel
    return os.path.join(root, "images", rel)

def load_task(name, root, n=None, seed=42):
    
    cfg = TASKS[name]
    raw = json.load(open(os.path.join(root, "data", cfg["file"])))
    imgs = anns = None
    if cfg["metric"] == "caption":
        cc = json.load(open(os.path.join(root, "data", cfg["refs"])))
        imgs = {im["id"]: im["file_name"] for im in cc["images"]}
        anns = {}
        for a in cc["annotations"]:
            anns.setdefault(a["image_id"], []).append(a["caption"])
    rows = []
    for i, r in enumerate(raw):
        image = r.get("image") or None
        row = {"_idx": i, "id": str(r["question_id"]), "image": image,
               "prompt": make_question(cfg["q"], r["text"])}
        if cfg["metric"] == "caption":
            fn = imgs[i + 1]
            if os.path.basename(fn) != os.path.basename(image):
                raise ValueError(f"{name}: coco_type image {i + 1} is {fn}, expected {image}")
            row["target"] = anns[i + 1]
        elif cfg["metric"] == "iou":
            row["target"] = {"bbox": r["answer_bbox"], "size": r["size"]}
        elif cfg["metric"] == "letter":
            row["target"] = {"answer": str(r["answer"]), "image": image is not None}
        else:
            row["target"] = str(r["answer"])
        rows.append(row)
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows

def drop_missing_images(rows, root):
    
    kept, skipped = [], 0
    for r in rows:
        if r["image"] is None or os.path.exists(image_path(root, r["image"])):
            kept.append(r)
        else:
            skipped += 1
    return kept, skipped

def parse_letter(pred_text):
    
    if pred_text in OPTIONS:
        return pred_text
    if len(pred_text) >= 3 and pred_text[0] in OPTIONS and pred_text[1:3] == ". ":
        return pred_text[0]
    res = re.compile(r"The answer is ([A-Z]).").findall(pred_text)
    return res[0] if len(res) == 1 else "FAILED"

def change_bbox(bbox, im_w, im_h):
    
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    max_wh = max(im_w, im_h)
    if im_w > im_h:
        y1, y2 = y1 + (im_w - im_h) / 2, y2 + (im_w - im_h) / 2
    elif im_h > im_w:
        x1, x2 = x1 + (im_h - im_w) / 2, x2 + (im_h - im_w) / 2
    return [x1 / max_wh, y1 / max_wh, x2 / max_wh, y2 / max_wh]

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2
    inter = max(0, min(x2, x22) - max(x1, x21)) * max(0, min(y2, y22) - max(y1, y21))
    union = (x2 - x1) * (y2 - y1) + (x22 - x21) * (y22 - y21) - inter
    return inter / union if union > 0 else 0

def parse_bbox_upstream(text):
    
    try:
        s = text.replace("[", "").replace("]", "")
        vals = [float(x) for x in s[1:-1].split(",")]
    except Exception:
        return None
    return vals if len(vals) == 4 else None

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")

def parse_bbox_clean(text):
    vals = [float(x) for x in _FLOAT_RE.findall(text)]
    return vals if len(vals) == 4 else None

def _iou_correct(pred, target, parser):
    bbox = parser(pred)
    if bbox is None:
        return 0, None
    gt = [float(x) for x in target["bbox"].replace("[", "").replace("]", "").split(",")]
    max_wh = max(target["size"])
    iou = calculate_iou([x * max_wh for x in bbox], [x * max_wh for x in gt])
    return int(iou > 0.5), iou

def caption_scores(outputs, refs):
    
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    gts = {i: [{"caption": c} for c in rs] for i, rs in enumerate(refs)}
    res = {i: [{"caption": o}] for i, o in enumerate(outputs)}
    tok = PTBTokenizer()
    gts, res = tok.tokenize(gts), tok.tokenize(res)
    out = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for k, v in enumerate(bleu, 1):
        out[f"Bleu_{k}"] = float(v) * 100.0
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        m, _ = Meteor().compute_score(gts, res)
        out["METEOR"] = m * 100.0
    except Exception as e:
        print(f"  METEOR unavailable ({str(e)[:80]}); averaging the other 6 metrics", flush=True)
        out["METEOR"] = None
    out["ROUGE_L"] = float(Rouge().compute_score(gts, res)[0]) * 100.0
    out["CIDEr"] = float(Cider().compute_score(gts, res)[0]) * 100.0
    vals = [v for v in out.values() if v is not None]
    out["score"] = sum(vals) / len(vals)
    return out

def _norm_answer(x):
    
    x = " ".join(str(x).strip().split()).lower()
    return x.strip("\"'\u201c\u201d\u2018\u2019.!? ")

def score_task(task, outputs, targets):
    
    metric = TASKS[task]["metric"]
    n = len(outputs)
    assert n == len(targets)
    rec = {"n": n, "n_empty": sum(1 for o in outputs if not o.strip())}
    if n == 0:
        rec["acc"] = None
        return rec
    if metric == "letter":
        letters = [parse_letter(o) for o in outputs]
        correct = [int(l == t["answer"]) for l, t in zip(letters, targets)]
        img = [c for c, t in zip(correct, targets) if t["image"]]
        rec.update(acc=sum(correct) / n, acc_img=(sum(img) / len(img) if img else None), n_img=len(img),
                   n_failed_parse=sum(1 for l in letters if l == "FAILED"), correct=correct)
    elif metric == "substring":
        correct = [int((t.upper() in o.upper()) or (o.upper() in t.upper())) for o, t in zip(outputs, targets)]
        rec.update(acc=sum(correct) / n, correct=correct)
    elif metric == "exact_upper":
        correct = [int(o.upper() == t.upper()) for o, t in zip(outputs, targets)]
        norm = [int(_norm_answer(o) == _norm_answer(t)) for o, t in zip(outputs, targets)]
        rec.update(acc=sum(correct) / n, acc_norm=sum(norm) / n, correct=correct)
    elif metric == "exact_lower_unans":
        correct = [int(("Unanswerable" not in o) and o.lower() == t.lower()) for o, t in zip(outputs, targets)]
        norm = [int(("Unanswerable" not in o) and _norm_answer(o) == _norm_answer(t)) for o, t in zip(outputs, targets)]
        rec.update(acc=sum(correct) / n, acc_norm=sum(norm) / n, n_unanswerable=sum(1 for o in outputs if "Unanswerable" in o), correct=correct)
    elif metric == "iou":
        up = [_iou_correct(o, t, parse_bbox_upstream) for o, t in zip(outputs, targets)]
        clean = [_iou_correct(o, t, parse_bbox_clean) for o, t in zip(outputs, targets)]
        ious = [i for _, i in clean if i is not None]
        rec.update(acc=sum(c for c, _ in up) / n, acc_fullparse=sum(c for c, _ in clean) / n,
                   n_unparsed=sum(1 for c, i in clean if i is None),
                   mean_iou=(sum(ious) / len(ious) if ious else None), correct=[c for c, _ in up])
    elif metric == "caption":
        cs = caption_scores(outputs, targets)
        rec.update(cs)
        rec["acc"] = cs["score"] / 100.0
    else:
        raise ValueError(metric)
    return rec

def model_name(spec):
    if spec == "base":
        return "base"
    kind, _, arg = spec.partition(":")
    if kind == "merged":
        return "merged_" + os.path.basename(os.path.normpath(arg))
    if kind == "client":
        return "client_" + arg
    raise ValueError(f"--model must be base | merged:<dir> | client:<Task>, got {spec!r}")

def adapter_stamp(spec, root):
    
    if spec == "base":
        return None
    kind, _, arg = spec.partition(":")
    f = (os.path.join(arg, "adapter_model.safetensors") if kind == "merged"
         else os.path.join(LP.adapter_dir(root, client_task(arg)), "adapter_model.bin"))
    if not os.path.exists(f):
        raise FileNotFoundError(f"adapter weights not found: {f}")
    return round(os.path.getmtime(f), 1)

def client_task(arg):
    
    if arg in LP.TASKS:
        return arg
    name = resolve_tasks(arg)[0]
    adapter = TASKS[name].get("adapter")
    if adapter is None:
        raise ValueError(f"{name} is an unseen task: no client adapter exists")
    return adapter

def load_model(spec, root, dtype="fp16", device="cuda"):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel
    from safetensors.torch import load_file
    kw = {"device_map": device}
    if dtype == "8bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                       llm_int8_skip_modules=["multi_modal_projector", "vision_tower", "lm_head"])
        kw["dtype"] = torch.float16
    elif dtype == "fp32":
        kw["dtype"] = torch.float32
    else:
        kw["dtype"] = torch.float16
    model = LlavaForConditionalGeneration.from_pretrained(LP.BASE_MODEL, **kw)
    processor = AutoProcessor.from_pretrained(LP.BASE_MODEL)
    processor.tokenizer.padding_side = "left"
    if spec != "base":
        kind, _, arg = spec.partition(":")
        if kind == "merged":
            model = PeftModel.from_pretrained(model, arg)
            pf = os.path.join(arg, "projector.safetensors")
            if os.path.exists(pf):
                LP.apply_projector(model, load_file(pf))
            else:
                print(f"  {arg}: no projector.safetensors -> keeping the base projector", flush=True)
        elif kind == "client":
            factors, proj, cfg = LP.load_llava_adapter(LP.adapter_dir(root, client_task(arg)))
            with tempfile.TemporaryDirectory() as tmp:
                LP.write_llava_merged(factors, tmp, cfg["r"])
                model = PeftModel.from_pretrained(model, tmp)
            if not proj:
                raise ValueError(f"client {arg}: adapter has no mm_projector in non_lora_trainables.bin")
            LP.apply_projector(model, proj)
        else:
            raise ValueError(spec)
    model.eval()
    return model, processor

def expand2square(pil_img, background_color):
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result

def load_image(path, pad, image_mean):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if pad:
        img = expand2square(img, tuple(int(x * 255) for x in image_mean))
    return img

def _model_dtype(model):
    
    for name, p in model.named_parameters():
        if "vision_tower" in name:
            return p.dtype
    return next(model.parameters()).dtype

def _gen_batch(model, processor, prompts, images, max_new_tokens):
    
    import torch
    tok = processor.tokenizer
    inputs = processor(images=images if images else None, text=prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(_model_dtype(model))
    oom = False
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if len(prompts) == 1 or not ("out of memory" in msg or isinstance(e, torch.cuda.OutOfMemoryError)):
            raise
        oom = True
    if oom:
        del inputs
        gc.collect(); torch.cuda.empty_cache()
        print(f"  OOM on batch of {len(prompts)}; retrying one at a time", flush=True)
        outs = []
        for i, p in enumerate(prompts):
            outs += _gen_batch(model, processor, [p], [images[i]] if images else [], max_new_tokens)
        return outs
    new = out[:, inputs["input_ids"].shape[1]:]
    return [tok.decode(row, skip_special_tokens=True).strip() for row in new]

def generate(model, processor, rows, cfg, root, batch_size):
    
    outs = [None] * len(rows)
    image_mean = processor.image_processor.image_mean
    for with_image in (True, False):
        idx = [i for i, r in enumerate(rows) if (r["image"] is not None) == with_image]
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            prompts = [build_prompt(rows[i]["prompt"], with_image) for i in chunk]
            images = ([load_image(image_path(root, rows[i]["image"]), cfg["pad"], image_mean) for i in chunk]
                      if with_image else [])
            for i, o in zip(chunk, _gen_batch(model, processor, prompts, images, cfg["max_new_tokens"])):
                outs[i] = o
    return outs

def metrics_path(out_dir, name):
    return os.path.join(out_dir, f"{name}__metrics.json")

def _write_metrics(out_dir, name, results, task):
    
    fn = metrics_path(out_dir, name)
    on_disk = json.load(open(fn)) if os.path.exists(fn) else {}
    on_disk[task] = results[task]
    results.clear(); results.update(on_disk)
    tmp = fn + ".tmp"
    with open(tmp, "w") as f:
        json.dump(on_disk, f, indent=1)
    os.replace(tmp, fn)

def _pending(name, task_names, out_dir, n, force, stamp, seed=42):
    
    fn = metrics_path(out_dir, name)
    results = json.load(open(fn)) if os.path.exists(fn) else {}
    if force:
        return results, list(task_names)
    todo, skipped = [], []
    for t in task_names:
        rec = results.get(t)
        done = (rec is not None and rec.get("n_requested") == n and rec.get("adapter_mtime") == stamp
                and rec.get("seed", seed) == seed and rec.get("n_skipped", 0) == 0)
        (skipped if done else todo).append(t)
    if skipped:
        print(f"[{name}] resume: skipping done tasks {skipped}", flush=True)
    return results, todo

def gen_cfg(task, dtype):
    c = TASKS[task]
    return {"prompt": PROMPT_VERSION, "max_new_tokens": c["max_new_tokens"], "pad": c["pad"],
            "greedy": True, "dtype": dtype}

def read_jsonl(fn):
    return [json.loads(l) for l in open(fn)]

def rescore(name, task_names, out_dir):
    
    results = json.load(open(metrics_path(out_dir, name))) if os.path.exists(metrics_path(out_dir, name)) else {}
    for t in task_names:
        fn = os.path.join(out_dir, f"{name}__{t}.jsonl")
        if not os.path.exists(fn):
            print(f"[{name}] {t}: no generations", flush=True)
            continue
        rows = read_jsonl(fn)
        rec = dict(results.get(t) or {"model": name, "task": t, "metric": TASKS[t]["metric"]})
        sc = score_task(t, [r["output"] for r in rows], [r["target"] for r in rows])
        sc.pop("correct", None)
        rec.update(sc)
        results[t] = rec
        _write_metrics(out_dir, name, results, t)
        print(f"[{name}] {t}: acc={rec['acc']} n={rec['n']}", flush=True)
    return results

def evaluate(spec, task_names, n, out_dir, root, seed=42, batch_size=4, dtype="fp16", device="cuda", force=False):
    name = model_name(spec)
    os.makedirs(out_dir, exist_ok=True)
    stamp = adapter_stamp(spec, root)
    results, todo = _pending(name, task_names, out_dir, n, force, stamp, seed=seed)
    if not todo:
        print(f"[{name}] nothing to do", flush=True)
        return results
    model, processor = load_model(spec, root, dtype=dtype, device=device)
    for t in todo:
        t0 = time.time()
        cfg = TASKS[t]
        rows = load_task(t, root, n=n, seed=seed)
        rows, skipped = drop_missing_images(rows, root)
        if skipped:
            print(f"[{name}] {t}: {skipped} example(s) skipped (missing image)", flush=True)
        if not rows:
            print(f"[{name}] {t}: no example has its image available -> not scored (download images first)", flush=True)
            continue
        outs = generate(model, processor, rows, cfg, root, batch_size)
        with open(os.path.join(out_dir, f"{name}__{t}.jsonl"), "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "idx": r["_idx"], "target": r["target"], "output": o,
                                    "has_image": r["image"] is not None}) + "\n")
        rec = {"model": name, "task": t, "metric": cfg["metric"], "n_requested": n, "n_skipped": skipped, "seed": seed,
               "seconds": round(time.time() - t0, 1), "adapter_mtime": stamp, "gen_cfg": gen_cfg(t, dtype)}
        sc = score_task(t, outs, [r["target"] for r in rows])
        sc.pop("correct", None)
        rec.update(sc)
        results[t] = rec
        _write_metrics(out_dir, name, results, t)
        acc = rec["acc"]
        print(f"[{name}] {t}: acc={acc if acc is None else round(acc, 4)} n={rec['n']} {rec['seconds']}s", flush=True)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=os.path.join(ROOT, "external", "mm_mergebench"))
    ap.add_argument("--out", default=res("lmm_eval"))
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "8bit", "fp32"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    tasks = resolve_tasks(args.tasks)
    if args.rescore:
        rescore(model_name(args.model), tasks, args.out)
        return
    evaluate(args.model, tasks, args.n, args.out, args.root, seed=args.seed, batch_size=args.batch_size,
             dtype=args.dtype, device=args.device, force=args.force)

if __name__ == "__main__":
    main()
