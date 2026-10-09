import json, os, sys, tempfile
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.pool import load_local_adapter_factors
from src.merge import (RULES, _delta, frame_similarity, chance_frame_similarity, frame_clusters,
                        regime_info, merge_regime_switch, merge_avg_factors, merged_rank)

D_IN, D_OUT, R = 256, 64, 8
CELLS = [(0, "q_proj"), (1, "k_proj")]

def _client(rng, A0=None, drift=0.3):
    
    fs = {}
    for c in CELLS:
        A = (A0[c] + drift * rng.standard_normal((R, D_IN)) * A0[c].std()) if A0 is not None \
            else rng.standard_normal((R, D_IN))
        fs[c] = {"A": A, "B": rng.standard_normal((D_OUT, R)) * 0.1}
    return fs

def _pools(seed=0):
    rng = np.random.default_rng(seed)
    A0 = {c: rng.standard_normal((R, D_IN)) for c in CELLS}
    shared = {f"s{i}": _client(rng, A0) for i in range(3)}
    indep = {f"x{i}": _client(rng) for i in range(2)}
    return shared, indep

def test_local_loader_folds_alpha_over_r_into_B_and_parses_peft_keys():
    import torch
    from safetensors.torch import save_file
    d = tempfile.mkdtemp()
    A = torch.randn(R, D_IN); B = torch.randn(D_OUT, R)
    save_file({"base_model.model.model.layers.3.self_attn.v_proj.lora_A.weight": A,
               "base_model.model.model.layers.3.self_attn.v_proj.lora_B.weight": B}, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"r": R, "lora_alpha": 16, "target_modules": ["v_proj"]}, open(os.path.join(d, "adapter_config.json"), "w"))
    fs, cfg = load_local_adapter_factors(d)
    assert list(fs) == [(3, "v_proj")] and cfg["lora_alpha"] == 16
    assert np.allclose(fs[(3, "v_proj")]["A"], A.numpy(), atol=1e-6)
    assert np.allclose(fs[(3, "v_proj")]["B"], 2.0 * B.numpy(), atol=1e-6)
    raw, _ = load_local_adapter_factors(d, fold_alpha=False)
    assert np.allclose(raw[(3, "v_proj")]["B"], B.numpy(), atol=1e-6)

def test_frame_similarity_is_one_for_identical_frames_and_chance_for_independent():
    shared, indep = _pools()
    names, S = frame_similarity({"a": shared["s0"], "b": shared["s0"]})
    assert np.allclose(S, 1.0)
    names, S = frame_similarity(indep)
    ch, sd = chance_frame_similarity(D_IN, R)
    assert abs(S[0, 1] - ch) < 5 * sd
    names, S = frame_similarity(shared)
    assert S[0, 1] > 0.8 and S[0, 2] > 0.8

def test_frame_clusters_group_shared_frame_clients_and_leave_independent_singletons():
    shared, indep = _pools()
    pool = {**shared, **indep}
    info = regime_info(pool, tau=0.5)
    assert info["clusters"] == [["s0", "s1", "s2"], ["x0"], ["x1"]]
    assert info["max_below"] < 0.5 < info["min_above"]
    assert info["chance_mean"] < 0.2
    names, S = frame_similarity(pool)
    for tau in (0.2, 0.5, 0.8):
        assert frame_clusters(names, S, tau) == info["clusters"]

def test_regime_switch_equals_factor_rule_on_all_shared_pool():
    shared, _ = _pools()
    rs = merge_regime_switch(shared)
    fa = merge_avg_factors(shared)
    for c in CELLS:
        assert np.allclose(rs[c]["A"], fa[c]["A"]) and np.allclose(rs[c]["B"], fa[c]["B"])
    assert merged_rank(rs) == R

def test_regime_switch_equals_product_rule_on_all_independent_pool():
    _, indep = _pools()
    indep["x2"] = _client(np.random.default_rng(7))
    rs = merge_regime_switch(indep, product="trim_g50")
    fn, kw = RULES["trim_g50"]
    pr = fn(indep, **kw)
    for c in CELLS:
        assert np.allclose(_delta(rs[c]), _delta(pr[c]), atol=1e-6)
    assert merged_rank(rs) == kw["rank"]

def test_regime_switch_mixed_pool_merges_cluster_factorwise_then_product_across():
    shared, indep = _pools()
    pool = {**shared, **indep}
    rs = merge_regime_switch(pool, product="concat")
    cluster = merge_avg_factors(shared)
    virtual = {"s": cluster, **indep}
    fn, kw = RULES["concat"]
    expect = fn(virtual, **kw)
    for c in CELLS:
        assert np.allclose(_delta(rs[c]), _delta(expect[c]), atol=1e-6)
    assert merged_rank(rs) == 3 * R

def test_written_merged_adapter_round_trips_through_local_loader_with_scale_one():
    from src.pool import write_merged_adapter
    _, indep = _pools()
    merged = merge_avg_factors(indep)
    d = tempfile.mkdtemp()
    write_merged_adapter(merged, d, R, description="test")
    back, cfg = load_local_adapter_factors(d)
    assert cfg["lora_alpha"] == cfg["r"] == R
    for c in CELLS:
        assert np.allclose(back[c]["A"], merged[c]["A"], rtol=2e-2, atol=1e-3)
        assert np.allclose(back[c]["B"], merged[c]["B"], rtol=2e-2, atol=1e-3)

def test_hub_loader_delegates_to_local_loader(monkeypatch):
    import torch
    from safetensors.torch import save_file
    from src import pool as P
    d = tempfile.mkdtemp()
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(R, D_IN),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(D_OUT, R)},
              os.path.join(d, "adapter_model.safetensors"))
    json.dump({"r": R, "lora_alpha": 32}, open(os.path.join(d, "adapter_config.json"), "w"))
    monkeypatch.setattr(P, "snapshot_download", lambda repo: d)
    hub, cfg = P.load_adapter_factors("sentiment")
    loc, _ = load_local_adapter_factors(d)
    assert np.allclose(hub[(0, "q_proj")]["B"], loc[(0, "q_proj")]["B"]) and cfg["lora_alpha"] == 32
    json.dump({"r": R, "lora_alpha": 32, "use_dora": True}, open(os.path.join(d, "adapter_config.json"), "w"))
    import pytest
    with pytest.raises(ValueError, match="sentiment/r8"):
        P.load_adapter_factors("sentiment")

def test_regime_switch_registry_entry_runs_the_c2_product_rule():
    fn, kw = RULES["regime_switch"]
    assert fn is merge_regime_switch and kw == {"tau": 0.5, "product": "trim_g50"}
    assert RULES["trim_g50"][1]["keep"] == 0.2 and RULES["trim_g50"][1]["gamma"] == 0.5

def test_avg_factors_normmatch_matches_concat_update_norm_per_cell():
    from src.merge import merge_avg_factors_normmatch, merge_concat
    _, indep = _pools()
    nm = merge_avg_factors_normmatch(indep); cc = merge_concat(indep); fa = merge_avg_factors(indep)
    for c in CELLS:
        assert np.isclose(np.linalg.norm(_delta(nm[c])), np.linalg.norm(_delta(cc[c])), rtol=1e-4)
        assert np.allclose(nm[c]["A"], fa[c]["A"])

def test_cap_share_enforces_no_majority_without_amplifying():
    from src.merge import cap_share_pool, client_norms, merge_cap, RULES
    _, indep = _pools()
    pool = {**indep, "heavy": {c: {"A": indep["x0"][c]["A"], "B": indep["x0"][c]["B"] * 50.0} for c in CELLS}}
    capped = cap_share_pool(pool, share=0.5)
    for c in CELLS:
        before, after = client_norms(pool, c), client_norms(capped, c)
        tot = sum(after.values())
        assert max(after.values()) / tot <= 0.5 + 1e-6
        assert all(after[t] <= before[t] + 1e-9 for t in pool)
        assert all(abs(after[t] - before[t]) < 1e-9 for t in ("x0", "x1"))
        assert after["heavy"] < before["heavy"]
        assert abs(after["heavy"] / tot - 0.5) < 1e-6
    two = {t: indep[t] for t in ("x0", "x1")}
    from src.merge import cap_share_pool as _cap
    inf = _cap(two, share=1 / 3)
    for c in CELLS:
        a = client_norms(inf, c); assert min(a.values()) > 0 and abs(max(a.values()) / sum(a.values()) - 0.5) < 1e-6
    same = cap_share_pool(indep, share=0.9)
    for c in CELLS:
        assert all(np.allclose(same[t][c]["B"], indep[t][c]["B"]) for t in indep)
    fn, kw = RULES["concat_cap50"]; m = fn(pool, **kw)
    assert set(m) == set(CELLS)

def test_whiten_pool_unit_spectrum_common_norm_same_subspaces():
    from src.merge import whiten_pool, client_norms, _lowrank_svd, RULES
    _, indep = _pools()
    pool = {**indep, "heavy": {c: {"A": indep["x0"][c]["A"], "B": indep["x0"][c]["B"] * 50.0} for c in CELLS}}
    w = whiten_pool(pool)
    for c in CELLS:
        med = np.median(list(client_norms(pool, c).values()))
        for t in pool:
            U, s, V = _lowrank_svd(w[t][c]["B"], w[t][c]["A"])
            assert np.allclose(s, s[0])
            assert abs(np.linalg.norm(_delta(w[t][c])) - med) < 1e-4 * med
            U0, _, V0 = _lowrank_svd(pool[t][c]["B"], pool[t][c]["A"])
            assert abs(np.linalg.svd(U0.T @ U, compute_uv=False).min() - 1) < 1e-6
            assert abs(np.linalg.svd(V0.T @ V, compute_uv=False).min() - 1) < 1e-6
    fn, kw = RULES["concat_white"]; m = fn(pool, **kw); assert set(m) == set(CELLS)
    from src.merge import merged_rank
    assert merged_rank(m) == len(pool) * R
    for c in CELLS:
        assert w["heavy"][c]["A"].shape == (R, D_IN) and w["heavy"][c]["B"].shape == (D_OUT, R)
    for name in ("isocts_white", "isoc_white"):
        fn, kw = RULES[name]; m = fn(pool, **kw)
        assert all(np.isfinite(m[c]["A"]).all() and np.isfinite(m[c]["B"]).all() and np.abs(m[c]["B"]).max() > 0 for c in CELLS)

def test_norm_matched_isotropic_keeps_directions_at_task_arithmetic_norm():
    from src.merge import RULES, merge_concat, norm_match
    _, indep = _pools()
    pool = {**indep, "heavy": {c: {"A": indep["x0"][c]["A"], "B": indep["x0"][c]["B"] * 50.0} for c in CELLS}}
    fn, kw = RULES["isocts_neq_nm"]; nm = fn(pool, **kw)
    raw = RULES["isocts_neq"][0](pool, **RULES["isocts_neq"][1])
    ref = RULES["normeq_med"][0](pool, **RULES["normeq_med"][1])
    cc = merge_concat(pool)
    for c in CELLS:
        assert np.isclose(np.linalg.norm(_delta(nm[c])), np.linalg.norm(_delta(ref[c])), rtol=1e-4)
        assert not np.isclose(np.linalg.norm(_delta(nm[c])), np.linalg.norm(_delta(cc[c])), rtol=0.2)
        d_nm, d_raw = _delta(nm[c]).ravel(), _delta(raw[c]).ravel()
        assert abs(d_nm @ d_raw / (np.linalg.norm(d_nm) * np.linalg.norm(d_raw)) - 1) < 1e-6

def test_client_step_is_the_scaled_polar_factor():
    
    from src.merge import whiten_pool, client_norms, _lowrank_svd
    rng = np.random.default_rng(0)
    pool = {f"c{i}": {c: {"A": rng.standard_normal((R, D_IN)), "B": rng.standard_normal((D_OUT, R)) * 10.0 ** i}
                      for c in CELLS} for i in range(3)}
    w = whiten_pool(pool)
    for c in CELLS:
        m = float(np.median(list(client_norms(pool, c).values())))
        for t in pool:
            U, S, V = _lowrank_svd(pool[t][c]["B"], pool[t][c]["A"])
            Q, P = U @ V.T, V @ np.diag(S) @ V.T
            rel = lambda X, Y: np.linalg.norm(X - Y) / max(np.linalg.norm(Y), 1e-12)
            assert rel(_delta(pool[t][c]), Q @ P) < 1e-5
            assert np.allclose(P, P.T) and np.linalg.eigvalsh(P).min() > -1e-9
            assert rel(_delta(w[t][c]), (m / np.sqrt(len(S))) * Q) < 1e-5
            assert np.isclose(np.linalg.norm(_delta(pool[t][c])), np.linalg.norm(P), rtol=1e-4)
