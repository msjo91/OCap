import json, os, sys
import numpy as np
import pytest
import torch
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import lmm_pool as LP
from scripts.eval import lmm_eval as LE

BENCH_ROOT = os.path.join(ROOT, "external", "mm_mergebench")

def test_key_re_parses_real_adapter_key():
    m = LP.KEY_RE.search("base_model.model.model.layers.31.mlp.down_proj.lora_A.weight")
    assert (int(m.group(1)), m.group(2), m.group(3)) == (31, "mlp.down_proj", "A")
    m = LP.KEY_RE.search("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")
    assert m.groups() == ("0", "self_attn.q_proj", "B")
    m = LP.KEY_RE.search("base_model.model.model.language_model.layers.7.self_attn.o_proj.lora_A.weight")
    assert m.groups() == ("7", "self_attn.o_proj", "A")
    assert LP.KEY_RE.search("base_model.model.model.mm_projector.0.weight") is None
    assert LP.KEY_RE.search("base_model.model.model.layers.3.self_attn.q_proj.weight") is None

def test_proj_re_parses_projector_key():
    m = LP.PROJ_RE.search("base_model.model.model.mm_projector.2.bias")
    assert (LP.PROJ_MAP[m.group(1)], m.group(2)) == ("linear_2", "bias")
    assert LP.PROJ_RE.search("base_model.model.model.mm_projector.1.weight") is None

def test_target_modules_regex_selects_only_language_model():
    import re
    pat = re.compile(LP.TARGET_MODULES_RE)
    assert pat.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert pat.fullmatch("model.language_model.layers.31.mlp.down_proj")
    assert not pat.fullmatch("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not pat.fullmatch("model.multi_modal_projector.linear_1")

def _toy_factors(rank=4, layers=2, d_in=32, d_hidden=64, seed=0):
    rng = np.random.default_rng(seed)
    q = lambda *shape: rng.integers(-8, 9, size=shape).astype(np.float64) / 8
    dims = {"self_attn.q_proj": (d_in, d_in), "self_attn.k_proj": (d_in, d_in), "self_attn.v_proj": (d_in, d_in),
            "self_attn.o_proj": (d_in, d_in), "mlp.gate_proj": (d_in, d_hidden), "mlp.up_proj": (d_in, d_hidden),
            "mlp.down_proj": (d_hidden, d_in)}
    return {(L, m): {"A": q(rank, di), "B": q(do, rank)} for L in range(layers) for m, (di, do) in dims.items()}

def test_write_llava_merged_reload_roundtrip(tmp_path):
    factors = _toy_factors()
    proj = {"linear_1.weight": torch.randn(8, 4), "linear_1.bias": torch.randn(8),
            "linear_2.weight": torch.randn(8, 8), "linear_2.bias": torch.randn(8)}
    LP.write_llava_merged(factors, str(tmp_path), 4, projector=proj, description="toy")
    got, gproj = LP.read_llava_merged(str(tmp_path))
    assert set(got) == set(factors)
    for cell in factors:
        for f in ("A", "B"):
            assert np.array_equal(got[cell][f], factors[cell][f]), cell
    assert set(gproj) == set(proj) and all(torch.equal(gproj[k], proj[k]) for k in proj)
    cfg = json.load(open(tmp_path / "adapter_config.json"))
    assert cfg["r"] == cfg["lora_alpha"] == 4 and cfg["target_modules"] == LP.TARGET_MODULES_RE
    assert open(tmp_path / "MERGE_INFO.txt").read().strip() == "toy"

def _tiny_llava():
    from transformers import LlavaConfig, LlavaForConditionalGeneration, CLIPVisionConfig, LlamaConfig
    torch.manual_seed(0)
    vc = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                          image_size=32, patch_size=16, projection_dim=16)
    tc = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=2, vocab_size=64)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=63, vision_feature_layer=-1,
                      vision_feature_select_strategy="default", projector_hidden_act="gelu")
    return LlavaForConditionalGeneration(cfg)

def test_peft_loads_written_adapter_into_transformers_llava(tmp_path):
    from peft import PeftModel
    model = _tiny_llava()
    factors = _toy_factors()
    LP.write_llava_merged(factors, str(tmp_path), 4)
    base_q = model.get_submodule("model.language_model.layers.1.self_attn.q_proj")
    x = torch.randn(3, 32)
    base_out = base_q(x).detach()
    pm = PeftModel.from_pretrained(model, str(tmp_path))
    lora = [n for n, _ in pm.named_parameters() if "lora_" in n]
    assert len(lora) == 2 * 7 * 2 and not any("vision" in n for n in lora)
    wrapped = pm.get_submodule("base_model.model.model.language_model.layers.1.self_attn.q_proj")
    fs = factors[(1, "self_attn.q_proj")]
    delta = torch.tensor(fs["B"] @ fs["A"], dtype=torch.float32)
    torch.testing.assert_close(wrapped(x).detach() - base_out, x @ delta.T, atol=1e-4, rtol=1e-4)

def test_apply_projector_overwrites_wrapped_model(tmp_path):
    from peft import PeftModel
    model = _tiny_llava()
    LP.write_llava_merged(_toy_factors(), str(tmp_path), 4)
    pm = PeftModel.from_pretrained(model, str(tmp_path))
    mp = LP.projector_module(pm)
    proj = {"linear_1.weight": torch.randn_like(mp.linear_1.weight), "linear_2.bias": torch.randn_like(mp.linear_2.bias)}
    LP.apply_projector(pm, proj)
    assert torch.equal(mp.linear_1.weight.detach(), proj["linear_1.weight"])
    assert torch.equal(mp.linear_2.bias.detach(), proj["linear_2.bias"])

def test_mean_projector_is_client_mean():
    projs = {"a": {"linear_1.bias": torch.tensor([1.0, 2.0])}, "b": {"linear_1.bias": torch.tensor([3.0, 6.0])}}
    assert torch.equal(LP.mean_projector(projs)["linear_1.bias"], torch.tensor([2.0, 4.0]))

def test_prompt_matches_vicuna_v1():
    assert LE.build_prompt("Q?", True) == LE.SYSTEM + " USER: <image>\nQ? ASSISTANT:"
    assert LE.build_prompt("Q?", False) == LE.SYSTEM + " USER: Q? ASSISTANT:"
    assert LE.SYSTEM.startswith("A chat between a curious user") and LE.SYSTEM.endswith("user's questions.")

def test_question_construction_per_runner():
    sqa = "<image>\nWhich is it?\nA. x\nB. y\nAnswer with the option's letter from the given choices directly."
    assert LE.make_question("sqa", sqa) == sqa[len("<image>\n"):]
    t = "What is the object in the image?\nAnswer the question using a single word or phrase."
    assert LE.make_question("imagenet", t) == "What is the object in the image?\n" + LE.IMAGENET_TEMPLATE
    assert LE.IMAGENET_TEMPLATE.startswith("Choose an answer from the choices below: Doormat,") and \
        LE.IMAGENET_TEMPLATE.endswith("Bernese mountain dog.")
    assert LE.make_question("text", t) == t

def test_task_table_and_aliases():
    assert len(LE.TASKS) == 12 and len(LE.resolve_tasks("seen")) == 8 and len(LE.resolve_tasks("unseen")) == 4
    assert LE.resolve_tasks("Grounding, flickr30k,Screen2Words,imagenet-r") == ["REC", "Flickr30k", "Screen2words", "ImageNet-R"]
    assert {t: c["max_new_tokens"] for t, c in LE.TASKS.items()} == {
        "ScienceQA": 1024, "ImageNet": 1024, "VQAv2": 1024, "REC": 1024, "OCRVQA": 128, "VizWiz": 128,
        "Flickr30k": 128, "IconQA": 128, "AOKVQA": 1024, "ImageNet-R": 1024, "Screen2words": 128, "TabMWP": 128}
    assert [t for t, c in LE.TASKS.items() if not c["pad"]] == ["ScienceQA", "ImageNet", "VQAv2", "REC", "AOKVQA"]
    assert LE.client_task("Flickr30k") == "flickr30k" and LE.client_task("REC") == "REC"
    with pytest.raises(ValueError):
        LE.client_task("AOKVQA")
    assert LE.model_name("merged:results/mm/merges/concat/") == "merged_concat"
    assert LE.model_name("client:VQAv2") == "client_VQAv2" and LE.model_name("base") == "base"

def test_score_letter_and_img_accuracy():
    outs = ["B", "B. apostrophe", "The answer is C.", "b", "", "A B", "D"]
    tg = [{"answer": "B", "image": True}, {"answer": "B", "image": False}, {"answer": "C", "image": True},
          {"answer": "B", "image": True}, {"answer": "A", "image": False}, {"answer": "A", "image": True},
          {"answer": "E", "image": False}]
    r = LE.score_task("ScienceQA", outs, tg)
    assert r["correct"] == [1, 1, 1, 0, 0, 0, 0]
    assert r["acc"] == pytest.approx(3 / 7) and r["acc_img"] == pytest.approx(2 / 4) and r["n_img"] == 4
    assert r["n_failed_parse"] == 3 and r["n_empty"] == 1
    assert LE.parse_letter("The answer is B. The answer is C.") == "FAILED"

def test_score_substring_either_direction():
    r = LE.score_task("ImageNet", ["a golden retriever dog", "Retriever", "cat", ""], ["Golden retriever"] * 4)
    assert r["correct"] == [1, 1, 0, 1] and r["n_empty"] == 1
    assert LE.score_task("ImageNet-R", ["Koala."], ["Koala"])["acc"] == 1.0

def test_score_exact_upper_and_lower():
    assert LE.score_task("VQAv2", ["Kite", "kites"], ["kite", "kite"])["correct"] == [1, 0]
    assert LE.score_task("IconQA", ["0", "0."], ["0", "0"])["correct"] == [1, 0]
    assert LE.score_task("TabMWP", ["18"], ["18"])["acc"] == 1.0 and LE.score_task("AOKVQA", ["3"], ["3"])["acc"] == 1.0
    r = LE.score_task("OCRVQA", ["the book", "Unanswerable", "The Book."], ["The Book", "The Book", "The Book."])
    assert r["correct"] == [1, 0, 1] and r["n_unanswerable"] == 1

def test_change_bbox_matches_benchmark_padded_square_convention():
    got = LE.change_bbox([280.45, 165.39, 81.98, 182.65], 427, 640)
    assert np.allclose(np.round(got, 2), [0.6, 0.26, 0.73, 0.54])
    assert LE.change_bbox([10, 20, 30, 40], 100, 100) == [0.1, 0.2, 0.4, 0.6]
    assert LE.change_bbox([0, 0, 100, 50], 100, 50) == [0.0, 0.25, 1.0, 0.75]

def test_score_iou_reproduces_upstream_parse_quirk():
    tg = {"bbox": "[0.6,0.26,0.73,0.54]", "size": [427, 640]}
    assert LE.parse_bbox_upstream("[0.6,0.26,0.73,0.54]") == [0.6, 0.26, 0.73, 0.5]
    assert LE.parse_bbox_clean("[0.6,0.26,0.73,0.54]") == [0.6, 0.26, 0.73, 0.54]
    r = LE.score_task("REC", ["[0.6,0.26,0.73,0.54]", "[0.1,0.1,0.2,0.2]", "no box", "[0.6, 0.26, 0.73, 0.541]"],
                      [tg] * 4)
    assert r["correct"] == [1, 0, 0, 1] and r["acc"] == 0.5 and r["acc_fullparse"] == 0.5 and r["n_unparsed"] == 1
    assert r["mean_iou"] == pytest.approx((1.0 + 0.0 + 1.0 / 1.0 * (0.13 * 0.28) / (0.13 * 0.281)) / 3, abs=1e-6)
    assert LE.calculate_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)

def test_score_caption_wiring():
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")
    refs = [["a man riding a horse", "a person on a horse"], ["two dogs playing in the grass"]]
    exact = LE.score_task("Flickr30k", ["A man riding a horse.", "Two dogs playing in the grass"], refs)
    off = LE.score_task("VizWiz", ["a red car", "an empty street"], refs)
    keys = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "score", "acc"}
    assert keys <= set(exact) and "correct" not in exact
    assert exact["Bleu_1"] == pytest.approx(100.0) and exact["ROUGE_L"] == pytest.approx(100.0)
    assert exact["score"] > off["score"] and exact["acc"] == pytest.approx(exact["score"] / 100)
    vals = [exact[k] for k in ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr") if exact[k] is not None]
    assert exact["score"] == pytest.approx(sum(vals) / len(vals))
    assert all(isinstance(exact[k], float) for k in keys if exact[k] is not None)
    if exact["METEOR"] is None:
        pytest.skip("METEOR unavailable (Java); averaged over 6 metrics")

def _fake_root(tmp_path, n_rows=6, n_images=4):
    from PIL import Image
    root = tmp_path / "bench"
    (root / "data" / "Seen_data" / "VQAv2").mkdir(parents=True)
    (root / "images" / "COCO2014" / "val2014").mkdir(parents=True)
    rows = [{"question_id": 100 + i, "image": f"./COCO2014/val2014/img{i}.jpg",
             "text": f"Q{i}?\nAnswer the question using a single word or phrase.", "answer": f"a{i}"} for i in range(n_rows)]
    json.dump(rows, open(root / "data" / "Seen_data" / "VQAv2" / "val.json", "w"))
    for i in range(n_images):
        Image.new("RGB", (8, 6), (255, 0, 0)).save(root / "images" / "COCO2014" / "val2014" / f"img{i}.jpg")
    return str(root)

def test_load_task_subsample_deterministic_and_missing_images_skipped(tmp_path):
    root = _fake_root(tmp_path)
    a = LE.load_task("VQAv2", root, n=5, seed=42)
    b = LE.load_task("VQAv2", root, n=5, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 5
    assert [r["id"] for r in LE.load_task("VQAv2", root, n=5, seed=7)] != [r["id"] for r in a]
    full = LE.load_task("VQAv2", root)
    assert len(full) == 6 and full[0]["target"] == "a0" and full[0]["prompt"].startswith("Q0?")
    assert LE.image_path(root, full[0]["image"]).endswith("bench/images/COCO2014/val2014/img0.jpg")
    kept, skipped = LE.drop_missing_images(full, root)
    assert skipped == 2 and [r["id"] for r in kept] == ["100", "101", "102", "103"]

def test_load_real_benchmark_rows_if_present():
    if not os.path.exists(os.path.join(BENCH_ROOT, "data", "Seen_data", "ScienceQA", "test.json")):
        pytest.skip("benchmark JSONs not downloaded")
    rows = LE.load_task("ScienceQA", BENCH_ROOT, n=50, seed=42)
    assert len(rows) == 50 and all(r["target"]["answer"] in LE.OPTIONS for r in rows)
    assert all("<image>" not in r["prompt"] for r in rows)
    rec = LE.load_task("REC", BENCH_ROOT, n=5, seed=42)[0]
    assert set(rec["target"]) == {"bbox", "size"}
    cap = LE.load_task("Flickr30k", BENCH_ROOT, n=5, seed=42)[0]
    assert isinstance(cap["target"], list) and len(cap["target"]) == 5

def test_expand2square_pads_to_mean_colour():
    from PIL import Image
    img = Image.new("RGB", (10, 4), (1, 2, 3))
    out = LE.expand2square(img, (7, 7, 7))
    assert out.size == (10, 10) and out.getpixel((0, 0)) == (7, 7, 7) and out.getpixel((5, 5)) == (1, 2, 3)
    assert LE.expand2square(Image.new("RGB", (4, 4)), (0, 0, 0)).size == (4, 4)

def test_pending_resume_and_atomic_metrics(tmp_path):
    out = str(tmp_path)
    results = {"VQAv2": {"n_requested": 5, "adapter_mtime": None, "acc": 0.5}}
    LE._write_metrics(out, "base", results, "VQAv2")
    assert not os.path.exists(os.path.join(out, "base__metrics.json.tmp"))
    _, todo = LE._pending("base", ["VQAv2", "IconQA"], out, 5, False, None)
    assert todo == ["IconQA"]
    assert LE._pending("base", ["VQAv2"], out, 10, False, None)[1] == ["VQAv2"]
    assert LE._pending("base", ["VQAv2"], out, 5, False, 123.4)[1] == ["VQAv2"]
    assert LE._pending("base", ["VQAv2"], out, 5, True, None)[1] == ["VQAv2"]
    other = json.load(open(os.path.join(out, "base__metrics.json"))); other["IconQA"] = {"acc": 1.0}
    json.dump(other, open(os.path.join(out, "base__metrics.json"), "w"))
    LE._write_metrics(out, "base", {"VQAv2": {"acc": 0.7}}, "VQAv2")
    on_disk = json.load(open(os.path.join(out, "base__metrics.json")))
    assert on_disk["IconQA"] == {"acc": 1.0} and on_disk["VQAv2"]["acc"] == 0.7

def test_rescore_from_jsonl(tmp_path):
    out = str(tmp_path)
    with open(os.path.join(out, "base__VQAv2.jsonl"), "w") as f:
        for o, t in [("kite", "kite"), ("ball", "kite")]:
            f.write(json.dumps({"id": "x", "target": t, "output": o, "has_image": True}) + "\n")
    res = LE.rescore("base", ["VQAv2"], out)
    assert res["VQAv2"]["acc"] == 0.5 and res["VQAv2"]["n"] == 2
    assert json.load(open(os.path.join(out, "base__metrics.json")))["VQAv2"]["acc"] == 0.5

def test_merge_rules_run_on_rectangular_mlp_cells():
    
    import numpy as np
    from src.merge import RULES, merged_rank, _delta
    rng = np.random.default_rng(0); r = 4
    cells = [(0, "mlp.down_proj"), (0, "mlp.up_proj"), (0, "self_attn.q_proj")]
    dims = {"mlp.down_proj": (48, 96), "mlp.up_proj": (96, 48), "self_attn.q_proj": (48, 48)}
    pool = {f"c{i}": {c: {"A": rng.standard_normal((r, dims[c[1]][1])), "B": rng.standard_normal((dims[c[1]][0], r)) * (10 ** (i - 1))}
                      for c in cells} for i in range(3)}
    for rule in ("concat", "trim_g50", "isocts_g50", "isocts_neq", "ties64", "dare64"):
        fn, kw = RULES[rule]
        kw = {**kw, "rank": 12} if "rank" in kw else kw
        m = fn(pool, **kw)
        for c in cells:
            d = _delta(m[c]); assert d.shape == dims[c[1]] and np.isfinite(d).all(), (rule, c)
        assert merged_rank(m) in (12, 3 * r), rule

def test_exact_match_reports_punctuation_normalized_variant():
    r = LE.score_task("VQAv2", ["No.", "yes", "Top wire."], ["no", "YES", "top wire"])
    assert r["acc"] == pytest.approx(1 / 3) and r["acc_norm"] == pytest.approx(1.0)
    r = LE.score_task("OCRVQA", ["Yes.", "Unanswerable."], ["Yes", "Unanswerable"])
    assert r["acc"] == 0.0 and r["acc_norm"] == pytest.approx(0.5)
