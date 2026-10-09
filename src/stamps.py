GEN_CFG = "v2:max_length=8192,trunc=left"
TASKS_AFFECTED_BY_V2 = {"xbrl_tags_extract", "xbrl_value_extract", "xbrl_formula_extract",
                        "xbrl_formula_calc_extract"}
SCORER_VERSION = "s3:hybrid-boundary,longest-tie,numeric(value,formula,float-rows)"

HELDOUT_CUE = {"finqa": "Answer:", "convfinqa": "A:", "ectsum": "Summary:", "finexam10k": "Answer:(appended)"}

def style_suffix(prompt_style):
    
    return "" if prompt_style in (None, "raw") else f",style={prompt_style}"

def pool_gen_cfg(prompt_style="raw"):
    return GEN_CFG + style_suffix(prompt_style)

def heldout_gen_cfg(cfg, task=None, prompt_style="raw"):
    
    cue = HELDOUT_CUE.get(task, "single")
    return (f"v2:max_length={cfg['max_length']},max_new={cfg['max_new_tokens']},trunc=left,cue={cue}"
            + style_suffix(prompt_style))

def heldout_record_is_current(rec, task, cfg):
    
    if rec is None:
        return False
    want = heldout_gen_cfg(cfg, task)
    got = rec.get("gen_cfg")
    if got == want:
        return True
    if got and got.endswith("cue=single") and task in ("finqa", "convfinqa"):
        return got.replace("cue=single", f"cue={HELDOUT_CUE[task]}") == want
    return False

def generation_is_current(rec, task, want, suite):
    
    if rec is None:
        return False
    if suite == "heldout":
        from .heldout import TASKS as HT
        return heldout_record_is_current(rec, task, HT[task])
    legacy_ok = None if task in TASKS_AFFECTED_BY_V2 else want
    return rec.get("gen_cfg", legacy_ok) == want
