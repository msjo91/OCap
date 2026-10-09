import json, os, sys, tempfile
import numpy as np
import pytest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import tasks as T, stamps as S
from src.pool import peft_scale
from src.merge import normalize_pool, clip_cells_pool, merge_ties, merge_concat, _delta

SENT = ["positive", "neutral", "negative"]
TAGS = ["us-gaap:Assets", "us-gaap:AssetsCurrent", "us-gaap:Liabilities"]

needs_finlora = pytest.mark.skipif(not os.path.isdir(T.FINLORA_DATA),
                                   reason="FinLoRA data missing; run scripts/data/fetch_external.sh")
needs_heldout = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ROOT, "data", "heldout", t, "test.jsonl"))
            for t in ("finqa", "convfinqa", "finexam10k")),
    reason="held-out data missing; run scripts/data/ingest_heldout.py")

def test_no_is_not_credited_inside_not():
    assert T.first_occurrence_pred("0\nExplanation: the headline does not talk", ["Yes", "No"]) is None

def test_longest_label_wins_position_tie():
    assert T.first_occurrence_pred("us-gaap:AssetsCurrent.", TAGS) == "us-gaap:AssetsCurrent"

def test_prefix_of_longer_non_inventory_token_not_credited():
    assert T.first_occurrence_pred("us-gaap:LiabilitiesAndStockholdersEquity", TAGS) is None

def test_label_soup_and_junk_suffix_still_credited_like_finlora():
    assert T.first_occurrence_pred("neutralpositive positive", SENT) == "neutral"
    assert T.first_occurrence_pred("neutral_REFNEER", SENT) == "neutral"
    assert T.first_occurrence_pred("No.", ["Yes", "No"]) == "No"

def test_earliest_occurrence():
    assert T.first_occurrence_pred("negative? no: positive", SENT) == "negative"

def test_value_extract_numeric_ignores_trailing_dot_zero():
    assert T.score_task("xbrl_value_extract", ["14004000000", "$14,004,000,000.0"], ["14004000000.0"] * 2)["acc"] == 1.0

def test_value_extract_does_not_credit_prefix_number():
    assert T.score_task("xbrl_value_extract", ["0.14"], ["0.1"])["acc"] == 0.0

def test_formula_percent_is_literal_text():
    assert T.score_task("formula", ["20.00%"], ["20.0"])["acc"] == 1.0
    assert T.score_task("formula", ["40,000.00."], ["40000.0"])["acc"] == 1.0

def test_mixed_inventory_float_rows_compared_numerically():
    lab = TAGS + ["4718000000.0"]
    rec = T.score_task("xbrl_tags_extract", ["4718000000", "us-gaap:AssetsCurrent"], ["4718000000.0", "us-gaap:AssetsCurrent"], lab)
    assert rec["acc"] == 1.0 and rec["n"] == 2

def test_legacy_record_is_wildcard_for_unaffected_pool_task():
    assert S.generation_is_current({"acc": 1}, "fpb", S.GEN_CFG, "pool")

def test_legacy_record_is_stale_for_long_prompt_tasks_and_heldout():
    from src import heldout as H
    for t in S.TASKS_AFFECTED_BY_V2:
        assert not S.generation_is_current({"acc": 1}, t, S.GEN_CFG, "pool")
    assert not S.generation_is_current({"acc": 1}, "finqa", None, "heldout")
    cur = {"gen_cfg": S.heldout_gen_cfg(H.TASKS["finexam10k"], "finexam10k")}
    assert S.generation_is_current(cur, "finexam10k", None, "heldout")
    single = {"gen_cfg": S.heldout_gen_cfg(H.TASKS["finexam10k"], None)}
    assert not S.generation_is_current(single, "finexam10k", None, "heldout")
    single_q = {"gen_cfg": S.heldout_gen_cfg(H.TASKS["convfinqa"], None)}
    assert S.generation_is_current(single_q, "convfinqa", None, "heldout")

@needs_heldout
def test_heldout_prompts_have_exactly_one_answer_slot():
    from src import heldout as H
    for t, tail in [("finqa", "Answer:"), ("convfinqa", "A:"), ("finexam10k", "letter only.\nAnswer:")]:
        ctx = H.load_task(t, n=1)[0]["context"]
        assert ctx.endswith(tail), (t, ctx[-30:])
        assert ctx.count("\nAnswer:") <= 1 and not ctx.endswith("A:\nAnswer:")

def test_metrics_write_merges_instead_of_overwriting():
    from src import runner as R
    d = tempfile.mkdtemp()
    a = {"t1": {"acc": 1}}; R._write_metrics(d, "m", a, task="t1")
    b = {"t2": {"acc": 2}}; R._write_metrics(d, "m", b, task="t2")
    on_disk = json.load(open(os.path.join(d, "m__metrics.json")))
    assert set(on_disk) == {"t1", "t2"} and set(b) == {"t1", "t2"}

def test_peft_scale_vanilla_and_rslora():
    assert peft_scale({"lora_alpha": 32, "r": 8}) == 4.0
    assert peft_scale({"lora_alpha": 16, "r": 8}) == 2.0
    assert abs(peft_scale({"lora_alpha": 16, "r": 8, "use_rslora": True}) - 16 / 8 ** 0.5) < 1e-12

def _toy_pool(seed=0, n=4, d_out=32, d_in=48, r=4, norms=(0.1, 1.0, 10.0, 100.0)):
    rng = np.random.default_rng(seed); pool = {}
    for i, nm in enumerate(norms):
        A = rng.standard_normal((r, d_in)); B = rng.standard_normal((d_out, r))
        B *= nm / np.linalg.norm(B @ A)
        pool[f"c{i}"] = {(0, "q_proj"): {"A": A, "B": B}, (1, "k_proj"): {"A": A * 0.5, "B": B * 2}}
    return pool

def test_normalize_pool_gamma_zero_is_identity_and_gamma_one_equalizes():
    pool = _toy_pool(); cell = (0, "q_proj")
    same = normalize_pool(pool, "median", gamma=0.0)
    assert all(np.allclose(_delta(same[c][cell]), _delta(pool[c][cell])) for c in pool)
    eq = normalize_pool(pool, "median", gamma=1.0, max_up=None)
    norms = [np.linalg.norm(_delta(eq[c][cell])) for c in pool]
    assert np.allclose(norms, norms[0])

def test_normalize_pool_max_up_never_amplifies():
    pool = _toy_pool(); cell = (0, "q_proj")
    capped = normalize_pool(pool, "median", gamma=1.0, max_up=1.0)
    for c in pool:
        assert np.linalg.norm(_delta(capped[c][cell])) <= np.linalg.norm(_delta(pool[c][cell])) + 1e-9

def test_clip_cells_touches_only_outlier_cells():
    pool = _toy_pool(); clipped = clip_cells_pool(pool, c=1e9)
    for c in pool:
        for cell in pool[c]:
            assert np.allclose(_delta(clipped[c][cell]), _delta(pool[c][cell]))

def test_ties_non_disjoint_mean_is_scale_matched_not_over_n():
    pool = _toy_pool(norms=(1.0, 1.0, 1.0, 1.0)); cell = (0, "q_proj")
    dis = merge_ties(pool, rank=8, keep=0.2, elect=True, disjoint=True)
    plain = merge_ties(pool, rank=8, keep=0.2, elect=True, disjoint=False)
    nd, npl = np.linalg.norm(_delta(dis[cell])), np.linalg.norm(_delta(plain[cell]))
    assert 0.5 < npl / nd < 2.0

def test_ties_mag_pool_takes_votes_from_pool_and_magnitudes_from_mag_pool():
    pool = _toy_pool(norms=(1.0, 1.0, 1.0, 1.0)); cell = (0, "q_proj")
    mag = {c: {k: {"A": v["A"], "B": v["B"] * 3.0} for k, v in pool[c].items()} for c in pool}
    base = merge_ties(pool, rank=8, keep=0.2); split = merge_ties(pool, rank=8, keep=0.2, mag_pool=mag)
    assert np.allclose(_delta(split[cell]), 3.0 * _delta(base[cell]), atol=1e-6)

def test_concat_is_exact_uniform_mean():
    pool = _toy_pool(); cell = (0, "q_proj"); m = merge_concat(pool)
    exact = np.mean([_delta(pool[c][cell]) for c in pool], axis=0)
    assert np.allclose(_delta(m[cell]), exact, rtol=1e-5, atol=1e-6)

@pytest.mark.skipif(os.environ.get("FEDQ_SKIP_TOKENIZER") == "1", reason="tokenizer download")
@needs_finlora
def test_generate_keeps_question_tail_of_long_xbrl_prompts_and_sub_batches():
    import torch
    from transformers import AutoTokenizer
    from src import runner as R
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": "<|pad|>"})
    rows, _ = T.load_task("xbrl_value_extract", n=100)
    long_rows = [r for r in rows if len(tok(r["context"])["input_ids"]) > 4096][:3]
    assert long_rows, "expected long prompts in the seed-42 sample"
    seen = []
    class Stub:
        device = torch.device("cpu")
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
            seen.append(input_ids.shape[1])
            tail = tok.decode(input_ids[0][-12:], skip_special_tokens=True)
            assert "Answer:" in tail, tail
            return torch.cat([input_ids, torch.full((input_ids.shape[0], 1), tok.eos_token_id)], 1)
    outs = R.generate(Stub(), tok, [r["context"] for r in long_rows], 4, batch_size=8, max_length=8192)
    assert len(outs) == len(long_rows) and max(seen) > 4096

@needs_finlora
def test_clean_filter_only_shrinks_and_removes_known_contamination():
    import hashlib, json as _json, os as _os, re as _re
    full, _ = T.load_task("headline", n=None)
    clean, _ = T.load_task("headline", n=None, clean=True)
    assert 0 < len(clean) < len(full)
    ids = {r["_idx"] for r in clean}
    assert ids <= {r["_idx"] for r in full}
    norm = lambda s: hashlib.md5(_re.sub(r"[^a-z0-9 ]", "", _re.sub(r"\s+", " ", str(s).lower()).strip()).encode()).hexdigest()
    assert len({norm(r["context"]) for r in clean}) == len(clean)
    tr = T._train_contexts()
    assert not any(norm(r["context"]) in tr for r in clean)
    for t in ("xbrl_tags_extract", "formula", "financebench"):
        a, _ = T.load_task(t, n=None); b, _ = T.load_task(t, n=None, clean=True)
        assert len(a) == len(b)

class _StubTok:
    
    bos_token = "<|begin_of_text|>"
    chat_template = "stub"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return (self.bos_token + "<|start_header_id|>user<|end_header_id|>\n\n"
                + msgs[0]["content"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

@needs_finlora
def test_raw_prompt_style_is_the_context_byte_for_byte():
    from src.runner import apply_prompt_style
    ctx = T.load_task("fpb", n=2, seed=42)[0]
    got = apply_prompt_style(_StubTok(), [r["context"] for r in ctx], "raw")
    assert got == [r["context"] for r in ctx]

def test_chat_prompt_style_drops_the_template_bos_so_the_tokenizer_adds_exactly_one():
    
    from src.runner import apply_prompt_style
    out = apply_prompt_style(_StubTok(), ["Instruction: x\nAnswer: "], "chat")[0]
    assert not out.startswith(_StubTok.bos_token)
    assert _StubTok.bos_token not in out
    assert out.startswith("<|start_header_id|>user<|end_header_id|>")
    assert out.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "Instruction: x" in out

def test_chat_prompt_style_requires_a_chat_template():
    from src.runner import apply_prompt_style
    class NoTemplate:
        bos_token = "<s>"
        chat_template = None
    with pytest.raises(ValueError):
        apply_prompt_style(NoTemplate(), ["x"], "chat")
    with pytest.raises(ValueError):
        apply_prompt_style(_StubTok(), ["x"], "instruct")

def test_raw_stamps_are_unchanged_so_every_existing_record_stays_current():
    
    assert S.pool_gen_cfg("raw") == S.GEN_CFG
    assert S.pool_gen_cfg() == S.GEN_CFG
    from src.heldout import TASKS as HT
    for t, cfg in HT.items():
        assert S.heldout_gen_cfg(cfg, t, "raw") == S.heldout_gen_cfg(cfg, t)

def test_chat_stamps_are_distinct_from_raw():
    from src.heldout import TASKS as HT
    assert S.pool_gen_cfg("chat") == S.GEN_CFG + ",style=chat"
    assert S.pool_gen_cfg("chat") != S.pool_gen_cfg("raw")
    cfg = HT["finqa"]
    assert S.heldout_gen_cfg(cfg, "finqa", "chat").endswith(",style=chat")
    assert not S.generation_is_current({"gen_cfg": S.pool_gen_cfg("raw")}, "fpb",
                                       S.pool_gen_cfg("chat"), "pool")
