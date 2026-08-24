"""Numerical checks for "Collapsing Uncertainty: An Operator View of Second-Order
Belief" (Adam M. Oberman).  Ancillary file: every numerical claim the paper makes
about its running examples is reproduced here, and the script exits 0 iff every
self-check passes.  NumPy is the only hard dependency; matplotlib is used
headlessly if importable and its absence does not break the run.

    python coin_example.py

Setup
-----
A single coin is drawn once by a fair toss and then flipped forever. We never
see which coin it is, only the flips y_1, y_2, ... in {0, 1} (1 = heads). Two
candidates: coin L with head probability theta_L = 0.4 and coin H with
theta_H = 0.6, prior weights w_L = w_H = 0.5.

The stream is exchangeable but *not* i.i.d.: one coin governs all flips, so each
flip is also evidence about which coin was drawn, and the past informs the
future. That dependence is the whole point. It is carried by two moments of the
mixing law -- mean m = E[theta] and variance v = Var[theta]:

    m = 0.5,   v = 0.5*0.4^2 + 0.5*0.6^2 - 0.5^2 = 0.01.

Because each coin is a single bias, all of v is the between-coin (model) channel.

Two closed forms this file is built to exhibit
----------------------------------------------
1. One head, then predict the next flip:

       P(y_2 = 1 | y_1 = 1) = m + v/m = 0.5 + 0.01/0.5 = 0.52.

2. Symmetric coins at 0.5 +/- d (so m = 0.5, v = d^2), one head, then predict:

       m + v/m = 0.5 + 2 d^2   ->   d = 0.05, 0.10, 0.30  gives  0.505, 0.520, 0.680.

Three reasoners see the *same* flip stream ("same information, several agents"):

  - TypedReasoner    keeps the posterior over which coin: update-then-marginalize.
  - CollapsedReasoner froze the marginal predictive before any flip: marginalize-
                      then-freeze. This is the collapse operator C; update is a no-op.
  - BernoulliLearner  fits one coin Bern(phi) with a Beta prior: it counts its way
                      to the drawn coin's frequency but carries no coin identity,
                      so its long-horizon predictive is a single point mass (E4
                      makes the contrast precise via the mixture-level projection
                      phi = 0.5, the frozen value the collapsed reasoner reports).

E6 adds the coupled-coin pair of paper Section 2.6: coins A and B drawn together
as one of two pairs, every marginal 0.5, and one American head moves the
Brazilian forecast to m_B + Cov/m_A = 0.48 (Cov = -0.01), a signed cross-coin
transfer no per-coin or pooled scalar state reproduces.

E7 compares the two readings of the collapse operator on that pair example: the
streamwise one the paper uses (one report per stream, frozen at 0.50 on every
query) against the joint one (freeze the one-step table on {0,1}^2, which keeps
the covariance for exactly one step). E8 checks whether split conformal over the
realized stream repairs the vanishing block coverage of Proposition 2 -- it does,
and it stops working across streams.

Run `python coin_example.py` to execute E1-E8 (E5 offline via mocks) with all
self-checks; it exits 0 iff every assert passes.
"""

import numpy as np

# ---------------------------------------------------------------------------
# The object: two coins, fair prior. m and v are all the theory needs.
# ---------------------------------------------------------------------------
BIASES = np.array([0.4, 0.6])      # theta_L, theta_H
WEIGHTS = np.array([0.5, 0.5])     # w_L, w_H
M = float(WEIGHTS @ BIASES)                       # mixing mean  = 0.5
V = float(WEIGHTS @ BIASES**2 - M**2)             # mixing var   = 0.01


# ---------------------------------------------------------------------------
# Reasoners. Shared interface: predict() -> P(next=1); update(y) conditions on y.
# ---------------------------------------------------------------------------
class TypedReasoner:
    """Keeps the model channel: a posterior over *which coin*.

    Update-then-marginalize. Observing y reweights each coin by its likelihood
    (theta_k if y=1 else 1-theta_k); predicting marginalizes the coin out,
    sum_k w_k theta_k. Because the weights move with the data, the past informs
    the future -- this is the reasoner that respects exchangeability. We carry
    log-weights so long streams don't underflow.
    """

    def __init__(self, biases=BIASES, weights=WEIGHTS):
        self.biases = np.asarray(biases, float)
        self.logw = np.log(np.asarray(weights, float))

    def predict(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        return float(w @ self.biases)                 # marginalize the coin out

    def update(self, y):
        p = self.biases if y else 1.0 - self.biases   # per-coin likelihood of y
        self.logw = self.logw + np.log(p)             # posterior in log-space


class CollapsedReasoner:
    """The flat baseline (Result 3): the collapse operator C.

    Marginalize-then-freeze. It stored only the prior predictive sum_k w_k theta_k
    as a frozen i.i.d. law and threw the coin identity away *before* any flip
    arrived. An i.i.d. law ignores its own past, so update is a no-op and it is
    pinned at 0.5 forever, whatever it sees.
    """

    def __init__(self, biases=BIASES, weights=WEIGHTS):
        self.p = float(np.asarray(weights, float) @ np.asarray(biases, float))

    def predict(self):
        return self.p

    def update(self, y):
        pass                                          # i.i.d.: the past is irrelevant


class BernoulliLearner:
    """The Result 4 baseline: fit a single coin Bern(phi), Beta(a,b) prior.

    A *realistic* flat reasoner -- it does learn, but only a single bias, not a
    coin identity. On one realized stream it counts its way to the drawn coin's
    frequency; what it cannot represent is the two-coin structure, so its
    long-horizon predictive is a single point mass. E4 makes that bite, using
    the mixture-level projection phi = 0.5 (the frozen collapsed value), which
    is calibrated on the next flip yet wrong on the long-run frequency.
    """

    def __init__(self, a=1.0, b=1.0):
        self.a, self.b = float(a), float(b)

    def predict(self):
        return self.a / (self.a + self.b)             # posterior mean of phi

    def update(self, y):
        self.a += y
        self.b += 1 - y


class TypedPairReasoner:
    """E6: keeps the model channel over *pairs* of coins (paper Section 2.6).

    One posterior weight is shared by both coins, so a flip of either coin
    reweights the pair hypotheses and thereby moves the *other* coin's forecast
    through the mixing covariance. Update-then-marginalize, per coin; no
    per-coin or pooled scalar state can carry this signed cross-coin transfer.
    """

    def __init__(self, pairs=((0.6, 0.4), (0.4, 0.6)), coins=("A", "B"),
                 weights=(0.5, 0.5)):
        self.pairs = np.asarray(pairs, float)         # row = pair hypothesis, col = coin
        self.col = {c: j for j, c in enumerate(coins)}
        self.logw = np.log(np.asarray(weights, float))

    def predict(self, coin):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        return float(w @ self.pairs[:, self.col[coin]])   # marginalize the pair out

    def update(self, coin, y):
        th = self.pairs[:, self.col[coin]]            # per-pair likelihood of y
        self.logw = self.logw + np.log(th if y else 1.0 - th)


# ---------------------------------------------------------------------------
# Simulation helper: draw one true coin, then flip it n times.
# ---------------------------------------------------------------------------
def simulate_stream(rng, n, biases=BIASES, weights=WEIGHTS):
    """Draw a coin once (~weights), then n i.i.d. flips of *that* coin."""
    k = rng.choice(len(biases), p=weights)            # the coin, drawn once
    theta = biases[k]
    flips = (rng.random(n) < theta).astype(int)
    return k, float(theta), flips


# ---------------------------------------------------------------------------
# E1. One-head closed form: predict is 0.5, then 0.52 = m + v/m after a head.
# ---------------------------------------------------------------------------
def e1_one_head():
    r = TypedReasoner()
    assert abs(r.predict() - 0.5) < 1e-12, r.predict()
    r.update(1)
    target = M + V / M
    assert abs(r.predict() - target) < 1e-12, (r.predict(), target)
    print(f"E1  one-head update: predict = {r.predict():.12f}  (m + v/m = {target:.12f})  OK")


# ---------------------------------------------------------------------------
# E2. Trajectory: same stream to every reasoner; snapshot predict() at a few n.
# ---------------------------------------------------------------------------
def e2_trajectory(rng, n=200):
    k, theta, flips = simulate_stream(rng, n)
    reasoners = {
        "typed": TypedReasoner(),
        "collapsed": CollapsedReasoner(),
        "bernoulli": BernoulliLearner(),
    }
    marks = [0, 1, 2, 5, 10, 50, 200]
    rows = {m: {} for m in marks}
    for i in range(n + 1):
        if i in rows:
            for name, r in reasoners.items():
                rows[i][name] = r.predict()
        if i < n:
            y = int(flips[i])
            for r in reasoners.values():
                r.update(y)

    # On a SINGLE realized stream the Bernoulli learner just counts, so it too
    # converges to the drawn coin's frequency theta -- not to 0.5. (The "-> 0.5"
    # claim is the KL projection of the whole mixture, a horizon statement made
    # precise in E4, not the limit of one online path.) What still separates it
    # from the typed reasoner is structure: typed reaches theta by keeping the
    # coin identity, so its long-horizon predictive stays bimodal over coins;
    # the Bernoulli collapses to a single structureless point mass. Collapsed
    # never moves at all.
    print(f"E2  trajectory: true coin = {'H' if theta > 0.5 else 'L'} (theta = {theta}); "
          f"typed drifts to {theta}, collapsed stays 0.5, bernoulli also -> {theta} (by counting)")
    print(f"    {'n':>4}  {'typed':>8}  {'collapsed':>9}  {'bernoulli':>9}")
    for m in marks:
        if m > n:
            continue
        r = rows[m]
        print(f"    {m:>4}  {r['typed']:>8.4f}  {r['collapsed']:>9.4f}  {r['bernoulli']:>9.4f}")


# ---------------------------------------------------------------------------
# E3. Spread sweep: coins at 0.5 +/- d give one-head predict = 0.5 + 2 d^2.
# ---------------------------------------------------------------------------
def e3_spread_sweep():
    print("E3  spread sweep: one head, coins at 0.5 +/- d, predict should be 0.5 + 2 d^2")
    for d in (0.05, 0.10, 0.30):
        r = TypedReasoner(biases=[0.5 - d, 0.5 + d], weights=[0.5, 0.5])
        r.update(1)
        target = 0.5 + 2 * d**2
        got = r.predict()
        assert abs(got - target) < 1e-12, (d, got, target)
        print(f"    d = {d:.2f}:  predict = {got:.4f}  (0.5 + 2 d^2 = {target:.4f})  OK")


# ---------------------------------------------------------------------------
# E4. Horizon contrast: predictive distribution of the block average Y_bar_N.
#     Typed -> bimodal near 0.4/0.6; flat (phi=0.5) -> tight bump ~N^-1/2 at 0.5.
# ---------------------------------------------------------------------------
def e4_horizon(rng, N=1000, hist_len=40):
    # A short shared history so the typed posterior is non-trivial but not
    # collapsed; feed it to the typed reasoner.
    _, theta, hist = simulate_stream(rng, hist_len)
    typed = TypedReasoner()
    for y in hist:
        typed.update(int(y))

    # The flat predictive is centered at phi = m = 0.5, the KL projection of the
    # mixture onto a single Bernoulli (Result 4). Note this is *not* where a naive
    # BernoulliLearner that just counts one coin's flips would land -- that tracks
    # the coin's own frequency; the phi=0.5 value is the mixture-level projection,
    # the frozen long-run frequency the collapsed reasoner is committed to.
    phi = M

    # Typed predictive of Y_bar_N: posterior-weighted mixture of Binomial(N, theta_k)/N.
    w = np.exp(typed.logw - typed.logw.max()); w /= w.sum()
    ks = np.arange(N + 1)
    grid = ks / N
    from math import lgamma
    logC = np.array([lgamma(N + 1) - lgamma(kk + 1) - lgamma(N - kk + 1) for kk in ks])
    typed_pmf = np.zeros(N + 1)
    for wk, th in zip(w, typed.biases):               # bimodal: one bump per coin
        typed_pmf += wk * np.exp(logC + ks * np.log(th) + (N - ks) * np.log(1 - th))

    # Flat predictive: Y_bar_N ~ Normal(phi, phi(1-phi)/N), width ~ N^-1/2.
    half = 1.96 * np.sqrt(phi * (1 - phi) / N)
    lo, hi = phi - half, phi + half

    # Two robust facts, independent of which coin the (stochastic) history favors:
    #  (1) the flat 95% interval never covers the true long-run frequency -- both
    #      coins sit 0.1 away from 0.5, far outside a +-0.03 bump;
    #  (2) the typed predictive is nearly disjoint from that interval: it lives on
    #      0.4/0.6, so almost none of its mass falls where the flat one is sure.
    # (typed_mass_near_truth is reported, not asserted: on an unlucky short history
    #  the posterior can back the wrong coin, yet the predictive is still bimodal.)
    covers_truth = lo <= theta <= hi
    typed_mass_in_flat = float(typed_pmf[(grid >= lo) & (grid <= hi)].sum())
    typed_mass_near_truth = float(typed_pmf[np.abs(grid - theta) <= 0.02].sum())

    print(f"E4  horizon contrast (N = {N}): true long-run frequency = {theta}")
    print(f"    flat 95% interval about phi={phi:.3f}: [{lo:.4f}, {hi:.4f}]  "
          f"covers truth? {covers_truth}")
    print(f"    typed posterior over coins = {dict(zip(('L','H'), np.round(w,3)))}; "
          f"typed mass within 0.02 of truth = {typed_mass_near_truth:.3f}; "
          f"typed mass inside flat interval = {typed_mass_in_flat:.4f}")
    print("    -> calibrated on the next flip, miscalibrated on the next thousand.")
    assert not covers_truth, "flat interval should not cover the mixture's true frequency"
    assert typed_mass_in_flat < 0.05, typed_mass_in_flat

    _maybe_plot_horizon(grid, typed_pmf, phi, N, theta)


def _maybe_plot_horizon(grid, typed_pmf, phi, N, theta):
    """Headless plot of both predictive distributions, if matplotlib imports."""
    try:
        import matplotlib
        matplotlib.use("Agg")                          # headless; never opens a window
        import matplotlib.pyplot as plt
    except Exception:
        print("    (matplotlib unavailable -- skipping coin_horizon.png)")
        return
    sd = np.sqrt(0.25 / N)
    flat_pdf = np.exp(-0.5 * ((grid - phi) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, typed_pmf * N, label="typed (mixture over coins)", lw=2)
    ax.plot(grid, flat_pdf, label="flat (Bernoulli at phi=0.5)", lw=2)
    ax.axvline(theta, color="k", ls="--", lw=1, label=f"true frequency {theta}")
    ax.set(xlabel=r"block average $\bar Y_N$", ylabel="predictive density",
           title=f"Horizon contrast, N = {N}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("coin_horizon.png", dpi=120)
    plt.close(fig)
    print("    saved coin_horizon.png")


# ---------------------------------------------------------------------------
# E4b. Coverage of the frozen interval (the paper's Limit box): draw the coin
#      20,000 times, flip it N = 2000 times, and count how often Y_bar_N lands
#      in the flat 95% interval 0.5 +/- 1.96 sqrt(0.25/N). Prop 2: coverage -> 0.
# ---------------------------------------------------------------------------
def e4b_coverage(rng, N=2000, draws=20_000):
    thetas = np.where(rng.random(draws) < 0.5, *BIASES)   # one coin per draw
    ybar = rng.binomial(N, thetas) / N
    half = 1.96 * np.sqrt(M * (1 - M) / N)
    coverage = float(np.mean(np.abs(ybar - M) <= half))
    print(f"E4b Limit-box coverage, {draws} draws at N = {N}: flat interval "
          f"[{M - half:.4f}, {M + half:.4f}] covers Y_bar_N with frequency {coverage:.4f}")
    assert coverage == 0.0, coverage


# ---------------------------------------------------------------------------
# E5. LLM comparison harness (offline). Slope of LLM one-head update vs typed.
#
# An LLM prompt states the coins, so a model's answer depends on the biases as
# well as the history. Every predictor here therefore takes (history, biases),
# and `llm_predict` is the single real-model hook.
#
# The biases are NOT optional context. `e5_llm_slope` sweeps the spread while
# holding the history fixed at [1], so the biases are the only thing that varies
# across the three regression points. A hook that ignored them would be asked
# the same question three times, return the same number three times, and score
# slope ~ 0 -- reading as "the model is frozen" when it means "the harness never
# varied the input". Put them in the prompt.
# ---------------------------------------------------------------------------
# WIRE A REAL MODEL HERE: replace the body of `llm_predict` with a single API
# call that returns the model's stated P(next heads) given the flip history and
# the two coin biases. Do not add an API dependency here -- see freeze_test.py,
# which imports this module and holds the real-model harness.
def llm_predict(history, biases):
    """Stated P(next flip = heads) given the flip history. Stub -> override."""
    raise NotImplementedError("plug in a real model, or use a mock below")


def real_llm(history, biases):
    """Adapter: score the real model. Both arguments reach the prompt."""
    return llm_predict(history, biases)


def mock_llm_flat(history, biases):
    """Mock: 'flips are independent' -> pinned at 0.5. Should give slope ~ 0."""
    return 0.5


def mock_llm_typed(history, biases):
    """Mock: a perfect reasoner that knows the coins. Should give slope ~ 1."""
    r = TypedReasoner(biases=biases, weights=[0.5, 0.5])
    for y in history:
        r.update(int(y))
    return r.predict()


def e5_llm_slope(predict_fn):
    """Regress (LLM one-head update) on (typed one-head update) across E3 spreads.

    Each 'update' is predict-after-one-head minus 0.5. Slope ~ 1 => the model
    tracks typed Bayes; slope ~ 0 => it stays pinned at 0.5 ('flips independent').
    """
    typed_updates, llm_updates = [], []
    for d in (0.05, 0.10, 0.30):
        biases = [0.5 - d, 0.5 + d]
        t = TypedReasoner(biases=biases, weights=[0.5, 0.5]); t.update(1)
        typed_updates.append(t.predict() - 0.5)
        llm_updates.append(predict_fn([1], biases) - 0.5)   # same one-head history
    x = np.array(typed_updates); y = np.array(llm_updates)
    return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")


def e5_selfcheck():
    print("E5  LLM slope harness (offline mocks):")
    slope_flat = e5_llm_slope(mock_llm_flat)
    slope_typed = e5_llm_slope(mock_llm_typed)
    print(f"    mock 'flat'  (always 0.5)      -> slope = {slope_flat:.3f}  (expect ~0)")
    print(f"    mock 'typed' (typed answer)    -> slope = {slope_typed:.3f}  (expect ~1)")
    assert abs(slope_flat - 0.0) < 1e-9, slope_flat
    assert abs(slope_typed - 1.0) < 1e-9, slope_typed
    print("    (override llm_predict(), then score real_llm against the typed slope.)")


# ---------------------------------------------------------------------------
# E6. Coupled coins (paper Section 2.6; supplement "The Coupled-Coin Example").
#     Two pairs M1: (theta_A, theta_B) = (0.6, 0.4), M2: (0.4, 0.6), fair toss
#     between them; flips independent given the pair. Every marginal is 0.5, so
#     per-coin collapse gives two fair coins; the coupling is only in the pair
#     posterior. One American head must move the Brazilian forecast DOWN.
# ---------------------------------------------------------------------------
def e6_coupled_coins():
    pairs = np.array([(0.6, 0.4), (0.4, 0.6)])
    w0 = np.array([0.5, 0.5])
    mA, mB = w0 @ pairs                               # marginal means, both 0.5
    varA = float(w0 @ pairs[:, 0] ** 2 - mA**2)       # 0.01, the old v
    cov = float(w0 @ (pairs[:, 0] * pairs[:, 1]) - mA * mB)   # -0.01

    r = TypedPairReasoner(pairs=pairs, weights=w0)
    assert abs(r.predict("A") - 0.5) < 1e-12, r.predict("A")
    assert abs(r.predict("B") - 0.5) < 1e-12, r.predict("B")

    r.update("A", 1)                                  # one American head
    tgt_A = mA + varA / mA                            # 0.52: the old m + v/m
    tgt_B = mB + cov / mA                             # 0.48: the cross-coin transfer
    assert abs(r.predict("A") - tgt_A) < 1e-12, (r.predict("A"), tgt_A)
    assert abs(r.predict("B") - tgt_B) < 1e-12, (r.predict("B"), tgt_B)

    # Per-coin baseline: one Beta per coin. The Brazilian learner sees no
    # Brazilian flips, so its forecast cannot move on any prefix of American
    # flips -- the channel that would carry the transfer does not exist.
    per_B = BernoulliLearner()
    pooled = BernoulliLearner()                       # one Beta over ALL flips
    for y in (1, 0, 1, 1):                            # arbitrary American prefix
        pooled.update(y)
        assert abs(per_B.predict() - 0.5) < 1e-12, per_B.predict()

    # Pooled baseline: ignores coin identity, so an American head raises its
    # forecast for EVERY coin -- the Brazilian moves up, the wrong direction.
    pooled_head = BernoulliLearner()
    pooled_head.update(1)
    assert pooled_head.predict() > 0.5, pooled_head.predict()

    print("E6  coupled coins: pairs M1=(0.6,0.4), M2=(0.4,0.6); one American head")
    print(f"    typed pair:  P(A) = {r.predict('A'):.4f}  (m_A + Var_A/m_A = {tgt_A:.4f})")
    print(f"                 P(B) = {r.predict('B'):.4f}  (m_B + Cov/m_A = {tgt_B:.4f}, Cov = {cov:+.2f})")
    print(f"    per-coin baseline: P(B) = {per_B.predict():.4f}  (never moves: no Brazilian data)")
    print(f"    pooled baseline:   P(B) = {pooled_head.predict():.4f}  (moves up -- the wrong direction)")


# ---------------------------------------------------------------------------
# E7. The two collapse operators on the two-pairs example (paper Section 2.6).
#     The collapse definition fixes one outcome space, and the pairs example
#     admits two readings of it.
#       - JOINT collapse: read the pair as one compound outcome y_i = (a_i, b_i)
#         in {0,1}^2. The frozen one-step table keeps E_G[theta_A theta_B] = 0.24,
#         hence the mixing covariance, so it reproduces the typed cross-forecast
#         0.48 *within* a time index and freezes at 0.50 across time indices.
#       - STREAMWISE collapse: read the coins as two indexed streams. Each stream
#         keeps only its own one-step predictive, giving two independent fair
#         coins, frozen at 0.50 on every query.
#     The paper uses the streamwise operator: a reporting system emits one
#     number per stream. This experiment checks that the two disagree here.
# ---------------------------------------------------------------------------
def e7_two_collapses():
    pairs = np.array([(0.6, 0.4), (0.4, 0.6)])
    w0 = np.array([0.5, 0.5])
    mA, mB = w0 @ pairs                               # both 0.5
    e_AB = float(w0 @ (pairs[:, 0] * pairs[:, 1]))    # E_G[theta_A theta_B]
    cov = e_AB - mA * mB
    assert abs(e_AB - 0.24) < 1e-12, e_AB
    assert abs(cov - (-0.01)) < 1e-12, cov

    # One-step joint table on {0,1}^2, averaged over the two pairs.
    def joint(a, b):
        pa = np.where(a, pairs[:, 0], 1.0 - pairs[:, 0])
        pb = np.where(b, pairs[:, 1], 1.0 - pairs[:, 1])
        return float(w0 @ (pa * pb))

    p11, p10, p01, p00 = joint(1, 1), joint(1, 0), joint(0, 1), joint(0, 0)
    assert abs(p11 - 0.24) < 1e-12, p11
    assert abs(p10 - 0.26) < 1e-12, p10
    assert abs(p01 - 0.26) < 1e-12, p01
    assert abs(p00 - 0.24) < 1e-12, p00
    assert abs(p11 + p10 + p01 + p00 - 1.0) < 1e-12

    # Typed reasoner, after one American head: 0.48 for every Brazilian flip.
    r = TypedPairReasoner(pairs=pairs, weights=w0)
    r.update("A", 1)
    typed_B = r.predict("B")
    assert abs(typed_B - 0.48) < 1e-12, typed_B

    # Joint collapse. Same time index: condition the frozen table on a_1 = 1.
    joint_b1 = p11 / (p11 + p10)
    # Later time index: coordinates are independent under the frozen table, so
    # both of the next step's flips revert to their marginals.
    joint_b2 = mB
    joint_a2 = mA
    assert abs(joint_b1 - 0.48) < 1e-12, joint_b1
    assert abs(joint_b1 - typed_B) < 1e-12          # agrees with typed here
    assert abs(joint_b2 - 0.50) < 1e-12, joint_b2
    assert abs(joint_b2 - typed_B) > 0.01           # and disagrees here
    # ... including on the American coin, where the typed reasoner gives 0.52.
    typed_A = r.predict("A")
    assert abs(typed_A - 0.52) < 1e-12, typed_A
    assert abs(joint_a2 - 0.50) < 1e-12, joint_a2
    assert abs(joint_a2 - typed_A) > 0.01

    # Streamwise collapse: two independent fair coins, frozen on every query.
    stream_b1 = stream_b2 = mB
    assert abs(stream_b1 - 0.50) < 1e-12
    assert abs(stream_b2 - 0.50) < 1e-12

    print("E7  two collapses on the pairs, after one American head (a_1 = 1):")
    print(f"    one-step joint table: p(1,1) = {p11:.2f}, p(1,0) = p(0,1) = {p10:.2f},"
          f" p(0,0) = {p00:.2f}")
    print(f"    typed          P(b_1|a_1) = {typed_B:.4f}   P(b_2|a_1) = {typed_B:.4f}")
    print(f"    joint collapse P(b_1|a_1) = {joint_b1:.4f}   P(b_2|a_1) = {joint_b2:.4f}"
          f"   P(a_2|a_1) = {joint_a2:.4f}"
          f"   (matches typed within the step, frozen after it)")
    print(f"    streamwise     P(b_1|a_1) = {stream_b1:.4f}   P(b_2|a_1) = {stream_b2:.4f}"
          f"   (frozen on both)")


# ---------------------------------------------------------------------------
# E8. Does split conformal repair Proposition 2's block-coverage failure?
#     Proposition 2 freezes the collapsed reasoner at i.i.d. Bern(0.5), so its
#     95% interval for a block average is 0.5 +/- 1.96 sqrt(0.25/N) and covers
#     with frequency 0 (E4b).  Conformal instead calibrates on the REALIZED
#     stream: cut it into blocks of length N and take the order-statistic
#     interval of J calibration block averages.  Permuting whole blocks is a
#     permutation of coordinates, so under the exchangeable true process the
#     J+1 block averages are exchangeable -- exactly conformal's hypothesis --
#     and coverage should be at least 1 - alpha.  It comes out right
#     conditionally on the drawn coin too, because given theta the blocks are
#     i.i.d.; that is a property of this process, not a distribution-free
#     conditional guarantee.
#     The cross-stream case of Section 2.6 is where the repair runs out: to
#     forecast a Brazilian block having seen only American flips there are no
#     Brazilian calibration blocks, and calibrating on American ones puts the
#     interval on the wrong side of 0.5 under anti-alignment.
# ---------------------------------------------------------------------------
def _conformal_interval(cal, alpha):
    """Order-statistic (identity-score) interval from J calibration values,
    valid for the (J+1)st exchangeable value."""
    j = len(cal)
    s = np.sort(cal)
    lo_rank = int(np.floor(alpha / 2 * (j + 1))) - 1
    hi_rank = int(np.ceil((1 - alpha / 2) * (j + 1))) - 1
    lo = -np.inf if lo_rank < 0 else s[lo_rank]
    hi = np.inf if hi_rank > j - 1 else s[hi_rank]
    return lo, hi


def e8_conformal_repair(rng, N=2000, J=40, alpha=0.05, reps=4000):
    half = 1.96 * np.sqrt(M * (1 - M) / N)            # the frozen interval
    frozen, conformal, thetas = [], [], []
    for _ in range(reps):
        theta = rng.choice(BIASES)                    # the coin, drawn ONCE
        blocks = rng.binomial(N, theta, size=J + 1) / N
        cal, test = blocks[:J], blocks[J]
        frozen.append(M - half <= test <= M + half)
        lo, hi = _conformal_interval(cal, alpha)
        conformal.append(lo <= test <= hi)
        thetas.append(theta)
    frozen = np.array(frozen); conformal = np.array(conformal)
    thetas = np.array(thetas)

    # Cross-stream: calibrate on American blocks, predict a Brazilian block.
    pairs = np.array([(0.6, 0.4), (0.4, 0.6)])
    cross = []
    for _ in range(reps):
        tA, tB = pairs[rng.integers(2)]
        cal_A = rng.binomial(N, tA, size=J) / N
        lo, hi = _conformal_interval(cal_A, alpha)
        cross.append(lo <= rng.binomial(N, tB) / N <= hi)
    cross = np.array(cross)

    print(f"E8  split conformal vs the frozen interval, N = {N}, J = {J} blocks,"
          f" alpha = {alpha}, {reps} reps:")
    print(f"    frozen collapsed interval [{M-half:.4f}, {M+half:.4f}]"
          f"   coverage = {frozen.mean():.4f}")
    print(f"    conformal on the same stream's blocks           "
          f"   coverage = {conformal.mean():.4f}  (nominal {1-alpha})")
    for t in BIASES:
        sel = thetas == t
        print(f"      conditional on theta = {t}: {conformal[sel].mean():.4f}")
    print(f"    conformal across streams (calibrate American, predict Brazilian)"
          f"   coverage = {cross.mean():.4f}")

    assert frozen.mean() < 0.01, frozen.mean()
    assert conformal.mean() > 1 - alpha - 0.02, conformal.mean()
    for t in BIASES:                                  # right conditionally too
        assert conformal[thetas == t].mean() > 1 - alpha - 0.03
    assert cross.mean() < 0.01, cross.mean()          # and it runs out here


# ---------------------------------------------------------------------------
# TODO (out of scope, Result 5): the "test selection" sharpening variant
# (confirming vs discriminating probe). In the plain coin *every* flip is
# discriminating, so it needs the cipher / a multi-hypothesis setup, not this toy.
# TODO (E5): no real LLM API integration beyond the stub above -- wire llm_predict.
# ---------------------------------------------------------------------------


def demo():
    rng = np.random.default_rng(20260702)             # seeded: runs reproduce
    print("=" * 68)
    print("Mixture-coin toy: typed vs collapsed vs Bernoulli  (m=0.5, v=0.01)")
    print("=" * 68)
    e1_one_head()
    print()
    e2_trajectory(rng)
    print()
    e3_spread_sweep()
    print()
    e4_horizon(rng)
    print()
    e4b_coverage(rng)
    print()
    e5_selfcheck()
    print()
    e6_coupled_coins()
    print()
    e7_two_collapses()
    print()
    e8_conformal_repair(rng)
    print()
    print("All experiments passed.")


if __name__ == "__main__":
    demo()
