import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.merge import (RULES, rank_of, merged_rank, merge_concat, merge_ties, merge_tsvm, merge_isoc,
                        merge_isocts, merge_knots, _knots_align, _ties_combine, _lowrank_svd,
                        merge_dcmerge, dcmerge_smooth_pool, _dcmerge_smooth, _dcmerge_trim,
                        _orth, whiten_pool, client_norms)

D_IN, D_OUT, R = 256, 64, 8
CELLS = [(0, "q_proj"), (1, "k_proj")]

def _client(rng, scale=0.1):
    return {c: {"A": rng.standard_normal((R, D_IN)), "B": rng.standard_normal((D_OUT, R)) * scale} for c in CELLS}

def _pool(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return {f"t{i}": _client(rng, scale=0.1 * (i + 1)) for i in range(n)}

def _dw(fs):
    return np.asarray(fs["B"], dtype=np.float64) @ np.asarray(fs["A"], dtype=np.float64)

def _nonzero_sv(M, tol=1e-6):
    sv = np.linalg.svd(M, compute_uv=False)
    return sv[sv > tol * sv[0]]

def test_lowrank_svd_is_exact_and_drops_zero_directions():
    rng = np.random.default_rng(1)
    B, A = rng.standard_normal((D_OUT, R)), rng.standard_normal((R, D_IN))
    U, s, V = _lowrank_svd(B, A)
    assert np.allclose(U @ np.diag(s) @ V.T, B @ A) and len(s) == R
    assert np.allclose(U.T @ U, np.eye(R)) and np.allclose(V.T @ V, np.eye(R))
    assert np.allclose(s, _nonzero_sv(B @ A)) and np.all(np.diff(s) <= 0)
    B[:, -1] = B[:, 0]; A[-1] = A[0]
    assert len(_lowrank_svd(B, A)[1]) == R - 1

def test_ties_combine_reproduces_merge_ties_and_reduces_to_mean_when_nothing_is_trimmed():
    pool = _pool(); cell = CELLS[0]
    deltas = [_dw(pool[t][cell]) for t in sorted(pool)]
    got = _ties_combine(deltas, deltas, keep=0.2, elect=True, disjoint=True)
    ref = merge_ties(pool, rank=64, keep=0.2)[cell]
    assert np.allclose(_dw(ref), got, atol=1e-4)
    assert np.allclose(_ties_combine(deltas, deltas, keep=1.0, elect=False, disjoint=False),
                       np.mean(deltas, axis=0))

def test_knots_alignment_identity_U_Z_i_equals_delta_i():
    pool = _pool(); tasks = sorted(pool)
    for cell in CELLS:
        U, Zs = _knots_align(pool, cell, tasks)
        assert U.shape[1] == len(tasks) * R and np.allclose(U.T @ U, np.eye(U.shape[1]))
        for t, Z in zip(tasks, Zs):
            assert np.allclose(U @ Z, _dw(pool[t][cell]))
        cat = np.hstack([_dw(pool[t][cell]) for t in tasks])
        assert np.allclose(np.linalg.norm(np.hstack(Zs), axis=1), _nonzero_sv(cat))

def test_knots_ta_equals_concat():
    pool = _pool()
    ta = merge_knots(pool, rank=64, method="ta"); cc = merge_concat(pool)
    for c in CELLS:
        assert np.allclose(_dw(ta[c]), _dw(cc[c]), atol=1e-9)
    assert merged_rank(ta) == 64

def test_knots_ties_trims_in_the_aligned_basis_and_differs_from_entrywise_ties():
    pool = _pool(); cell = CELLS[0]
    kt = merge_knots(pool, rank=64, method="ties", keep=0.2)
    U, Zs = _knots_align(pool, cell, sorted(pool))
    expect = U @ _ties_combine(Zs, Zs, keep=0.2, elect=True, disjoint=True)
    assert np.allclose(_dw(kt[cell]), expect, atol=1e-9)
    assert not np.allclose(_dw(kt[cell]), _dw(merge_ties(pool, rank=64, keep=0.2)[cell]), atol=1e-3)
    plain = merge_knots(pool, rank=64, method="ties", keep=1.0, elect=False, disjoint=False)
    assert np.allclose(_dw(plain[cell]), _dw(merge_concat(pool)[cell]), atol=1e-9)

def test_tsvm_single_client_returns_that_client():
    pool = {"only": _pool(1)["t0"]}
    m = merge_tsvm(pool, rank=64)
    for c in CELLS:
        assert np.allclose(_dw(m[c]), _dw(pool["only"][c]))
        assert m[c]["A"].shape == (64, D_IN) and m[c]["B"].shape == (D_OUT, 64)

def test_tsvm_orthogonal_clients_sum_exactly():
    rng = np.random.default_rng(3)
    pool = {"p": {}, "q": {}}
    for c in CELLS:
        B1 = np.zeros((D_OUT, R)); B1[:D_OUT // 2] = rng.standard_normal((D_OUT // 2, R))
        B2 = np.zeros((D_OUT, R)); B2[D_OUT // 2:] = rng.standard_normal((D_OUT // 2, R))
        A1 = np.zeros((R, D_IN)); A1[:, :D_IN // 2] = rng.standard_normal((R, D_IN // 2))
        A2 = np.zeros((R, D_IN)); A2[:, D_IN // 2:] = rng.standard_normal((R, D_IN // 2))
        pool["p"][c] = {"A": A1, "B": B1}; pool["q"][c] = {"A": A2, "B": B2}
    m = merge_tsvm(pool, rank=64, alpha=1.0)
    for c in CELLS:
        assert np.allclose(_dw(m[c]), _dw(pool["p"][c]) + _dw(pool["q"][c]))
    half = merge_tsvm(pool, rank=64, alpha=0.5)
    assert np.allclose(_dw(half[CELLS[0]]), 0.5 * _dw(m[CELLS[0]]))

def test_tsvm_general_pool_has_whitened_bases_and_keeps_every_singular_value():
    pool = _pool(); cell = CELLS[0]
    m = merge_tsvm(pool, rank=64)
    sv = _nonzero_sv(_dw(m[cell]))
    expect = np.sort(np.concatenate([_nonzero_sv(_dw(pool[t][cell])) for t in pool]))[::-1]
    assert len(sv) == 3 * R and np.allclose(sv, expect)
    assert not np.allclose(_dw(m[cell]), sum(_dw(pool[t][cell]) for t in pool), atol=1e-3)

def test_isoc_flat_spectrum_with_rank_of_task_arithmetic_sum():
    pool = _pool(); cell = CELLS[0]
    m = merge_isoc(pool, rank=64)
    ta = sum(_dw(pool[t][cell]) for t in pool)
    sv = _nonzero_sv(_dw(m[cell]))
    assert len(sv) == len(_nonzero_sv(ta)) == 3 * R
    assert np.allclose(sv, sv[0], rtol=1e-4)
    assert np.isclose(sv[0], _nonzero_sv(ta).mean(), rtol=1e-6)
    U, _, _ = _lowrank_svd(m[cell]["B"], m[cell]["A"]); Ut, _, _ = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in pool]),
                                                                              np.vstack([pool[t][cell]["A"] for t in pool]))
    assert np.allclose(U @ U.T, Ut @ Ut.T)
    assert np.allclose(_dw(merge_isoc(pool, rank=64, alpha=2.0)[cell]), 2.0 * _dw(m[cell]))

def test_isocts_flat_spectrum_with_rank_kc_plus_n_s():
    pool = _pool(); cell = CELLS[0]; n = len(pool)
    kc, s = 12, 4
    m = merge_isocts(pool, rank=64, k_common=kc, s=s)
    sv = _nonzero_sv(_dw(m[cell]))
    assert len(sv) == kc + n * s
    assert np.allclose(sv, sv[0], rtol=1e-4)
    d = merge_isocts(pool, rank=24, k_common=kc)
    assert len(_nonzero_sv(_dw(d[cell]))) == kc + n * ((24 - kc) // n)
    Ut, _, _ = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in pool]), np.vstack([pool[t][cell]["A"] for t in pool]))
    Um, _, _ = _lowrank_svd(m[cell]["B"], m[cell]["A"])
    P = Um @ Um.T
    assert np.allclose(P @ Ut[:, :kc], Ut[:, :kc])

def test_isocts_with_no_task_specific_directions_is_isoc_on_the_common_subspace():
    pool = _pool(); cell = CELLS[0]
    full = merge_isocts(pool, rank=64, k_common=3 * R, s=0)
    assert np.allclose(_dw(full[cell]), _dw(merge_isoc(pool, rank=64)[cell]))

def _sv(fs):
    return _nonzero_sv(_dw(fs))

def test_dcmerge_energy_smoothing_flattens_each_task_spectrum_and_preserves_its_subspaces():
    
    pool = _pool()
    sm = dcmerge_smooth_pool(pool, smoothing="mean")
    for t in pool:
        for c in CELLS:
            s0, s1 = _sv(pool[t][c]), _sv(sm[t][c])
            assert len(s0) == len(s1) == R
            assert np.allclose(s1, s1[0], rtol=1e-8)
            assert np.isclose(s1[0], s0.mean(), rtol=1e-8)
            assert np.isclose(s1.sum(), s0.sum(), rtol=1e-8)
            U0, _, V0 = _lowrank_svd(pool[t][c]["B"], pool[t][c]["A"])
            U1, _, V1 = _lowrank_svd(sm[t][c]["B"], sm[t][c]["A"])
            assert np.allclose(U0 @ U0.T, U1 @ U1.T)
            assert np.allclose(V0 @ V0.T, V1 @ V1.T)
    assert np.allclose(_dcmerge_smooth(np.array([4.0, 2.0, 0.5]), "none"), [4.0, 2.0, 0.5])

def test_dcmerge_linear_smoothing_targets_rho_and_reduces_to_averaging_at_rho_one():
    
    pool = _pool()
    lin = dcmerge_smooth_pool(pool, smoothing="linear", rho=5.0)
    for t in pool:
        for c in CELLS:
            s0, s1 = _sv(pool[t][c]), _sv(lin[t][c])
            assert np.isclose(s1.sum(), s0.sum(), rtol=1e-8)
            assert np.isclose(s1[0] / s1[-1], min(5.0, s0[0] / s0[-1]), rtol=1e-6)
            assert np.allclose(np.diff(s1), np.diff(s1)[0], rtol=1e-6)
    one = dcmerge_smooth_pool(pool, smoothing="linear", rho=1.0)
    avg = dcmerge_smooth_pool(pool, smoothing="mean")
    for t in pool:
        for c in CELLS:
            assert np.allclose(_dw(one[t][c]), _dw(avg[t][c]))

def test_dcmerge_mean_smoothing_equals_whiten_pool_up_to_a_per_client_scale():
    
    pool = _pool(n=4, seed=7)
    dc = dcmerge_smooth_pool(pool, smoothing="mean")
    wh = whiten_pool(pool, target="median")
    for c in CELLS:
        tgt = float(np.median(list(client_norms(pool, c).values())))
        ks = []
        for t in pool:
            Ddc, Dwh = _dw(dc[t][c]), _dw(wh[t][c])
            k = np.linalg.norm(Ddc) / np.linalg.norm(Dwh)
            assert np.linalg.norm(Ddc - k * Dwh) <= 1e-9 * np.linalg.norm(Ddc)
            assert np.isclose(k, _sv(pool[t][c]).mean() * np.sqrt(R) / tgt, rtol=1e-6)
            ks.append(k)
        assert max(ks) / min(ks) > 2

def test_dcmerge_nosmooth_differs_from_dcmerge_and_from_linear_smoothing():
    pool = _pool()
    base = dict(rank=64, trim_percent=1.0)
    mean_ = merge_dcmerge(pool, smoothing="mean", **base)
    none_ = merge_dcmerge(pool, smoothing="none", **base)
    lin_ = merge_dcmerge(pool, smoothing="linear", rho=5.0, **base)
    for other in (none_, lin_):
        assert any(not np.allclose(_dw(mean_[c]), _dw(other[c]), atol=1e-6) for c in CELLS)
    fm, km = RULES["dcmerge"]; fn, kn = RULES["dcmerge_nosmooth"]
    assert any(not np.allclose(_dw(fm(pool, **km)[c]), _dw(fn(pool, **kn)[c]), atol=1e-6)
               for c in CELLS)

def test_dcmerge_merges_in_the_tsvm_cover_basis_and_masks_off_diagonal_blocks():
    
    pool = _pool(); cell = CELLS[0]; tasks = sorted(pool); n = len(tasks)
    Us, Vs = [], []
    for t in tasks:
        U, _, V = _lowrank_svd(pool[t][cell]["B"], pool[t][cell]["A"])
        Us.append(U); Vs.append(V)
    Ut, Vt = _orth(np.hstack(Us)), _orth(np.hstack(Vs))
    assert np.allclose(Ut.T @ Ut, np.eye(n * R)) and np.allclose(Vt.T @ Vt, np.eye(n * R))
    free = merge_dcmerge(pool, rank=64, smoothing="none", trim_percent=1.0,
                         aggregate="ta", mask=False)
    expect = (Ut @ Ut.T) @ np.mean([_dw(pool[t][cell]) for t in tasks], axis=0) @ (Vt @ Vt.T)
    assert np.allclose(_dw(free[cell]), expect, atol=1e-8)
    masked = merge_dcmerge(pool, rank=64, smoothing="none", trim_percent=1.0,
                           aggregate="ta", mask=True)
    C = Ut.T @ _dw(masked[cell]) @ Vt
    blk = np.zeros((n * R, n * R))
    for i in range(n):
        blk[i * R:(i + 1) * R, i * R:(i + 1) * R] = 1.0
    assert np.allclose(C * (1 - blk), 0, atol=1e-8) and np.linalg.norm(C * blk) > 0
    assert np.allclose(C, (Ut.T @ expect @ Vt) * blk, atol=1e-8)

def test_dcmerge_trim_keeps_the_paper_fraction_of_entries():
    rng = np.random.default_rng(5)
    M = rng.standard_normal((32, 32))
    assert int(np.count_nonzero(_dcmerge_trim([M], 0.25)[0])) == int(0.25 * M.size)
    assert int(np.count_nonzero(_dcmerge_trim([M], 1e-6)[0])) == 1
    assert np.allclose(_dcmerge_trim([M], 1.0)[0], M)

def test_dcmerge_shapes_rank_finiteness_rectangular_cells_and_a_50x_heavier_client():
    rng = np.random.default_rng(11)
    wide, tall = (0, "q_proj"), (1, "k_proj")
    pool = {}
    for i in range(4):
        sc = 0.1 * (50.0 if i == 2 else 1.0)
        pool[f"t{i}"] = {
            wide: {"A": rng.standard_normal((R, 256)), "B": rng.standard_normal((64, R)) * sc},
            tall: {"A": rng.standard_normal((R, 64)), "B": rng.standard_normal((256, R)) * sc},
        }
    for name in ("dcmerge", "dcmerge_linear", "dcmerge_nosmooth"):
        fn, kw = RULES[name]
        m = fn(pool, **kw)
        assert set(m) == {wide, tall} and merged_rank(m) == 64 == rank_of(name, 4)
        assert m[wide]["A"].shape == (64, 256) and m[wide]["B"].shape == (64, 64)
        assert m[tall]["A"].shape == (64, 64) and m[tall]["B"].shape == (256, 64)
        for c in (wide, tall):
            d = _dw(m[c])
            assert np.all(np.isfinite(d)) and np.linalg.norm(d) > 0
            assert len(_nonzero_sv(d)) <= 4 * R
    a1 = merge_dcmerge(pool, rank=64, alpha=1.0)
    a2 = merge_dcmerge(pool, rank=64, alpha=2.5)
    for c in (wide, tall):
        assert np.allclose(_dw(a2[c]), 2.5 * _dw(a1[c]))

def test_dcmerge_registry_entries_match_the_paper_defaults():
    for name in ("dcmerge", "dcmerge_linear", "dcmerge_nosmooth"):
        fn, kw = RULES[name]
        assert fn is merge_dcmerge and rank_of(name, 8) == 64
        assert kw["trim_percent"] == 1e-3 and kw["aggregate"] == "ties" and kw["mask"] is True
        assert kw["alpha"] == 1.0 and merged_rank(fn(_pool(), **kw)) == 64
    assert RULES["dcmerge"][1]["smoothing"] == "mean"
    assert RULES["dcmerge_linear"][1]["smoothing"] == "linear"
    assert RULES["dcmerge_linear"][1]["rho"] == 5.0
    assert RULES["dcmerge_nosmooth"][1]["smoothing"] == "none"

def test_registry_entries_and_ranks():
    for name in ("tsvm", "isoc", "isocts", "knots_ties", "knots_ta"):
        fn, kw = RULES[name]
        assert rank_of(name, 8) == 64
        m = fn(_pool(), **kw)
        assert merged_rank(m) == 64 and set(m) == set(CELLS)
    assert RULES["isocts"][1]["k_common"] + 8 * RULES["isocts"][1]["s"] == 64
    assert RULES["knots_ties"][1]["keep"] == RULES["ties64"][1]["keep"]

def test_cap_ratio_is_inert_below_tau_and_one_sided_above():
    import numpy as np
    from src.merge import cap_ratio_pool, client_norms, RULES, _delta
    rng = np.random.default_rng(0); r, d_out, d_in = 4, 32, 48
    cells = [(0, "q_proj")]
    mk = lambda s: {c: {"A": rng.standard_normal((r, d_in)), "B": rng.standard_normal((d_out, r)) * s} for c in cells}
    mild = {f"c{i}": mk(1.0 + 0.3 * i) for i in range(4)}
    same = cap_ratio_pool(mild, tau=3.0)
    for c in cells:
        assert all(np.allclose(same[t][c]["B"], mild[t][c]["B"]) for t in mild)
    dominated = {**mild, "heavy": mk(50.0)}
    capped = cap_ratio_pool(dominated, tau=3.0)
    for c in cells:
        before, after = client_norms(dominated, c), client_norms(capped, c)
        med = float(np.median(list(before.values())))
        assert after["heavy"] <= 3.0 * med * 1.001 and after["heavy"] < before["heavy"]
        assert all(abs(after[t] - before[t]) < 1e-9 for t in mild)
    fn, kw = RULES["adaptive"]; m = fn(dominated, **{**kw, "rank": 5 * r})
    assert all(np.isfinite(_delta(m[c])).all() and np.abs(m[c]["B"]).max() > 0 for c in cells)

def test_mad_threshold_adapts_to_the_pool_and_is_inert_when_homogeneous():
    import numpy as np
    from src.merge import mad_tau, cap_ratio_pool, client_norms
    tight = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    assert mad_tau(tight) > tight.max() / np.median(tight)
    heavy = np.array([1.0, 1.0, 1.1, 50.0, 300.0])
    t = mad_tau(heavy)
    assert t < heavy.max() / np.median(heavy) and t > 1.0
    rng = np.random.default_rng(0); r, d_out, d_in = 4, 32, 48; cells = [(0, "q_proj")]
    mk = lambda s: {c: {"A": rng.standard_normal((r, d_in)), "B": rng.standard_normal((d_out, r)) * s} for c in cells}
    homo = {f"c{i}": mk(1.0 + 0.1 * i) for i in range(5)}
    out = cap_ratio_pool(homo, tau="mad")
    for c in cells:
        assert all(np.allclose(out[t2][c]["B"], homo[t2][c]["B"]) for t2 in homo)
    dom = {**homo, "heavy": mk(100.0)}
    capped = cap_ratio_pool(dom, tau="mad")
    for c in cells:
        assert client_norms(capped, c)["heavy"] < client_norms(dom, c)["heavy"]

def _dominated_pool(seed=0, r=4, d_out=32, d_in=48):
    rng = np.random.default_rng(seed); cells = [(0, "q_proj"), (1, "v_proj")]
    mk = lambda s: {c: {"A": rng.standard_normal((r, d_in)), "B": rng.standard_normal((d_out, r)) * s} for c in cells}
    return {**{f"c{i}": mk(1.0 + 0.1 * i) for i in range(5)}, "heavy": mk(100.0)}, cells

def test_ocap_rules_apply_the_operator_to_the_capped_pool_and_keep_every_direction():
    from src.merge import cap_ratio_pool, _delta, _BASE_RULES
    pool, cells = _dominated_pool()
    capped = cap_ratio_pool(pool, tau="mad")
    for c in cells:
        for t in pool:
            assert np.allclose(capped[t][c]["A"], pool[t][c]["A"])
            s = np.linalg.norm(capped[t][c]["B"]) / np.linalg.norm(pool[t][c]["B"])
            assert np.allclose(capped[t][c]["B"], pool[t][c]["B"] * s) and s <= 1.0 + 1e-12
        before, after = client_norms(pool, c), client_norms(capped, c)
        assert after["heavy"] < before["heavy"]
        assert all(abs(after[t] - before[t]) < 1e-9 for t in pool if t != "heavy")
    fn, kw = RULES["capmad_concat"]
    got, want = fn(pool, **kw), merge_concat(capped)
    assert all(np.allclose(_delta(got[c]), _delta(want[c])) for c in cells)
    for name in ("capmad_dcmerge", "capmad_isocts", "capmad_isoc", "capmad_tsvm", "capmad_knots", "capmad_ties"):
        fn, kw = RULES[name]
        assert kw["tau"] == "mad" and kw["base"] in _BASE_RULES
        m = fn(pool, **{**kw, "rank": 6 * 4})
        assert set(m) == set(cells) and all(np.isfinite(_delta(m[c])).all() for c in cells)

def test_median_clip_bounds_every_client_at_the_median_and_leaves_the_rest():
    from src.merge import cap_ratio_pool
    pool, cells = _dominated_pool(seed=1)
    fn, kw = RULES["capmed_concat"]
    assert kw == {"tau": 1.0, "base": "concat"}
    clipped = cap_ratio_pool(pool, tau=1.0)
    for c in cells:
        before, after = client_norms(pool, c), client_norms(clipped, c)
        med = float(np.median(list(before.values())))
        assert all(after[t] <= med * (1 + 1e-6) for t in pool)
        assert all(abs(after[t] - before[t]) < 1e-9 for t in pool if before[t] <= med)
        assert sum(after[t] < before[t] - 1e-9 for t in pool) == sum(before[t] > med for t in pool)

def test_norm_matched_control_has_the_ocap_norm_and_the_task_arithmetic_direction():
    from src.merge import _delta
    pool, cells = _dominated_pool(seed=2)
    fn, kw = RULES["concat_nm_capmad"]
    assert kw == {"rule": "concat", "ref": "capmad_concat"}
    ctrl, ta = fn(pool, **kw), merge_concat(pool)
    ocap = RULES["capmad_concat"][0](pool, **RULES["capmad_concat"][1])
    for c in cells:
        x, y, z = _delta(ctrl[c]), _delta(ta[c]), _delta(ocap[c])
        assert np.isclose(np.linalg.norm(x), np.linalg.norm(z))
        assert np.isclose(np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y)), 1.0)
        assert not np.allclose(x, z)
