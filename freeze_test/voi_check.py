"""Verify the value-of-information arithmetic for llm-uncertainty-via-design.md.

Setting: two-coin mixture (paper running example), G = 1/2 delta_{0.4} + 1/2 delta_{0.6},
m = 0.5, v = 0.01; coupled coins with Cov_G = -0.01.
All checks exact enumeration vs claimed closed forms.
"""

from itertools import product

TOL = 1e-12


def check(name, got, want):
    ok = abs(got - want) < TOL
    print(f"{'OK ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    assert ok, name


# ---------- single mixture coin ----------
def single_coin(thetas, weights):
    m = sum(w * t for w, t in zip(weights, thetas))
    v = sum(w * t * t for w, t in zip(weights, thetas)) - m * m
    return m, v


def brier_voi_single(thetas, weights):
    """Task: forecast y2 under squared (Brier) loss.
    Answer now: forecast m, expected loss E[(m - y2)^2].
    Observe y1 first (cost-free here): forecast E[y2|y1], expected loss E[Var-ish].
    VoI = loss(now) - loss(after)."""
    m, v = single_coin(thetas, weights)
    # enumerate joint law of (y1, y2)
    joint = {}
    for y1, y2 in product([0, 1], repeat=2):
        joint[(y1, y2)] = sum(
            w * (t if y1 else 1 - t) * (t if y2 else 1 - t)
            for w, t in zip(weights, thetas)
        )
    loss_now = sum(p * (m - y2) ** 2 for (y1, y2), p in joint.items())
    # optimal forecast after seeing y1 is conditional mean
    loss_after = 0.0
    for y1 in [0, 1]:
        p1 = sum(joint[(y1, y2)] for y2 in [0, 1])
        cond_mean = joint[(y1, 1)] / p1
        loss_after += sum(joint[(y1, y2)] * (cond_mean - y2) ** 2 for y2 in [0, 1])
    return loss_now - loss_after, m, v


# claimed: VoI_Brier = v^2 / (m(1-m))
voi, m, v = brier_voi_single([0.4, 0.6], [0.5, 0.5])
check("two-coin Brier VoI (enumeration)", voi, 0.0004)
check("two-coin Brier VoI closed form v^2/(m(1-m))", v * v / (m * (1 - m)), voi)

# conditional forecasts m + v/m and m - v/(1-m)
jt = {
    (y1, y2): sum(
        w * (t if y1 else 1 - t) * (t if y2 else 1 - t)
        for w, t in zip([0.5, 0.5], [0.4, 0.6])
    )
    for y1, y2 in product([0, 1], repeat=2)
}
check("P(y2=1|y1=1) = m + v/m", jt[(1, 1)] / (jt[(1, 0)] + jt[(1, 1)]), 0.52)
check("P(y2=1|y1=0) = m - v/(1-m)", jt[(0, 1)] / (jt[(0, 0)] + jt[(0, 1)]), 0.48)

# asymmetric sanity check of the closed form: coins 0.3 / 0.6, equal weights
voi2, m2, v2 = brier_voi_single([0.3, 0.6], [0.5, 0.5])
check("asymmetric Brier VoI closed form", v2 * v2 / (m2 * (1 - m2)), voi2)


# ---------- 0-1 world identification ----------
def id_voi(thetas, weights):
    """Task: declare which coin, utility 1 if correct.
    Answer now: max_k w_k. After one flip: E[max_k w_k(y1)]."""
    u_now = max(weights)
    u_after = 0.0
    for y1 in [0, 1]:
        # joint P(y1, k)
        jk = [w * (t if y1 else 1 - t) for w, t in zip(weights, thetas)]
        u_after += max(jk)  # sum_y max_k P(y,k) = E[max posterior]
    return u_after - u_now


voi_id = id_voi([0.4, 0.6], [0.5, 0.5])
check("two-coin 0-1 identification VoI", voi_id, 0.1)
# claimed closed form for equal weights, coins m -/+ d: VoI = d, i.e. sqrt(v)
check("identification VoI = sqrt(v) = d", voi_id, v ** 0.5)
# and E[max posterior] = 1/2 + d = 0.6
check("post-flip identification accuracy", 0.5 + voi_id, 0.6)

# asymmetric-bias sanity: coins 0.3/0.6 equal weights -> VoI = d = 0.15
check("asymmetric identification VoI = d", id_voi([0.3, 0.6], [0.5, 0.5]), 0.15)


# ---------- coupled coins ----------
# pairs: M1 (thA, thB) = (0.6, 0.4), M2 = (0.4, 0.6), weights 1/2
pairs = [(0.6, 0.4), (0.4, 0.6)]
wts = [0.5, 0.5]
mA = sum(w * tA for w, (tA, tB) in zip(wts, pairs))
mB = sum(w * tB for w, (tA, tB) in zip(wts, pairs))
cov = sum(w * tA * tB for w, (tA, tB) in zip(wts, pairs)) - mA * mB
check("Cov_G(thA, thB)", cov, -0.01)

# cross forecasts
jab = {
    (a1, b1): sum(
        w * (tA if a1 else 1 - tA) * (tB if b1 else 1 - tB)
        for w, (tA, tB) in zip(wts, pairs)
    )
    for a1, b1 in product([0, 1], repeat=2)
}
check("P(b1=1|a1=1) = mB + Cov/mA", jab[(1, 1)] / (jab[(1, 0)] + jab[(1, 1)]), 0.48)
check("P(b1=1|a1=0) = mB - Cov/(1-mA)", jab[(0, 1)] / (jab[(0, 0)] + jab[(0, 1)]), 0.52)

# Brier VoI of one American flip for forecasting the Brazilian flip
loss_now = sum(p * (mB - b1) ** 2 for (a1, b1), p in jab.items())
loss_after = 0.0
for a1 in [0, 1]:
    pa = jab[(a1, 0)] + jab[(a1, 1)]
    cm = jab[(a1, 1)] / pa
    loss_after += sum(jab[(a1, b1)] * (cm - b1) ** 2 for b1 in [0, 1])
voi_cross = loss_now - loss_after
check("cross-coin Brier VoI (enumeration)", voi_cross, 0.0004)
check("cross VoI closed form Cov^2/(mA(1-mA))", cov * cov / (mA * (1 - mA)), voi_cross)

# identification VoI from one American flip (which pair): same structure as single coin
voi_id_pair = id_voi([tA for tA, tB in pairs], wts)
check("which-pair 0-1 VoI from one American flip", voi_id_pair, 0.1)

print("\nAll checks passed.")


# ---------- decision-inert witness: J > 0 but identification VoI = 0 ----------
# Lopsided prior w = (0.9, 0.1) on coins (0.6, 0.4): every flip is
# forecast-moving and discriminating (J > 0), but no single flip changes the
# argmax declaration, so the 0-1 identification VoI is zero.
voi_lopsided = id_voi([0.6, 0.4], [0.9, 0.1])
check("lopsided-prior identification VoI is zero", voi_lopsided, 0.0)
# sanity: J > 0 there (single-outcome predictives differ across the models)
J_flip = sum((t1 - t2) ** 2 for t1, t2 in [(0.6, 0.4), (0.4, 0.6)])
assert J_flip > 0
# and the flip IS forecast-moving for the lopsided belief
m_lop = 0.9 * 0.6 + 0.1 * 0.4
post_head = (0.9 * 0.6) / m_lop  # weight on the 0.6 coin after a head
pred_after_head = post_head * 0.6 + (1 - post_head) * 0.4
assert abs(pred_after_head - m_lop) > 1e-6, "flip should be forecast-moving"
print("decision-inert witness verified: J>0, forecast-moving, VoI_id = 0")
