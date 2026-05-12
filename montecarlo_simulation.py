# %%
import numpy as np
import math
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt


# %% [markdown]
# Parametri del modello

# %%
@dataclass
class Params:
    # Static-game / substitution
    Delta: float = 1.0          # Δ
    alpha: float = 1.0          # α
    N: float = 1.0              # baseline customers per provider (scale)

    # Growth incentives (two types)
    K_G: float = 0.10
    K_P: float = 0.00

    # Noise in switching
    sigma: float = 0.025        # std of eps

    # Welfare weights
    eta: float = 1.0
    mu: float = 0.5
    lam: float = 1.0            # λ (avoid naming conflict with python lambda)
    rho: float = 1.0

    # Death process
    delta_G: float = 0.01
    delta_P: float = 0.03
    gamma_pi: float = 1.0
    gamma_c: float = 0.5

    # Birth process
    lambda_B: float = 10.0      # Poisson entry rate
    q_G: float = 0.2            # prob entrant is large

    # Bounds on markups (optional)
    M_min: float = -np.inf
    M_max: float = np.inf


def clip(x, lo, hi):
    return min(max(x, lo), hi)


# %% [markdown]
# Funzioni matematiche: 
# A(n), g(s;σ), NE, SO

# %%
def A_of_n(n: int, p: Params) -> float:
    # guard: n must be >=2 for log and meaningful competition
    n_eff = max(n, 2)
    return p.Delta * p.alpha * math.log(n_eff)


# %%
def normal_cdf(x: float) -> float:
    # standard normal CDF using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def g_abs_shifted_normal(s: float, sigma: float) -> float:
    """
    g(s; sigma) = E|s + eps|, eps ~ N(0, sigma^2), s >= 0
    """
    s = abs(s)
    if sigma <= 0:
        return s
    z = s / sigma
    # phi tail term with exp
    term1 = sigma * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z)
    term2 = s * (1.0 - 2.0 * normal_cdf(-z))
    return term1 + term2


# %%
def M_NE(n: int, p: Params):
    """
    Returns (M_G_NE, M_P_NE) for given total n
    """
    A = A_of_n(n, p)
    # guard against A==0 (very small n); A_of_n already enforces n>=2
    M_G = p.N / A - p.K_G
    M_P = p.N / A - p.K_P
    # optional bounds
    M_G = clip(M_G, p.M_min, p.M_max)
    M_P = clip(M_P, p.M_min, p.M_max)
    return M_G, M_P


# %%
def M_SO(n: int, p: Params) -> float:
    """
    M^SO(n) = (eta*n*N - mu) / (rho*n)
    """
    n_eff = max(n, 1)
    M = (p.eta * n_eff * p.N - p.mu) / (p.rho * n_eff)
    return clip(M, p.M_min, p.M_max)


# %% [markdown]
# Switching deterministico per tipo: 
# sg, sp
# 
# Usiamo la formula near-symmetric:

# %%
def s_type(nG: int, nP: int, p: Params):
    n = nG + nP
    if n <= 1:
        return 0.0, 0.0
    A = A_of_n(n, p)
    dK = p.K_G - p.K_P
    sG = A * (nP / (n - 1.0)) * dK
    sP = A * (nG / (n - 1.0)) * dK
    # These are magnitudes for churn; sign not needed for |.| quantities
    return sG, sP


# %% [markdown]
# Profitti, churn, welfare, PoA come funzioni di ng, np
# viene inoltre accettata la doppia versione stabile o feedback
# 

# %%
def per_provider_profit(nG: int, nP: int, p: Params, profit_mode: str = "feedback"):
    """
    Returns (pi_G, pi_P) expected profit per provider of each type.
    profit_mode: "stable" or "feedback"
    """
    n = nG + nP
    MG, MP = M_NE(n, p)
    sG, sP = s_type(nG, nP, p)

    if profit_mode == "stable":
        piG = p.N * MG
        piP = p.N * MP
    elif profit_mode == "feedback":
        piG = MG * (p.N + sG)
        piP = MP * (p.N + sP)
    else:
        raise ValueError("profit_mode must be 'stable' or 'feedback'")
    return piG, piP


def per_provider_churn(nG: int, nP: int, p: Params):
    """
    Returns (c_G, c_P) where c_t = E|s_t + eps|
    """
    sG, sP = s_type(nG, nP, p)
    cG = g_abs_shifted_normal(sG, p.sigma)
    cP = g_abs_shifted_normal(sP, p.sigma)
    return cG, cP


def welfare_SO(nG: int, nP: int, p: Params):
    """
    Expected welfare at SO for given (nG,nP). Depends on n only.
    Uses churn baseline n*E|eps|.
    """
    n = nG + nP
    n_eff = max(n, 1)
    M = M_SO(n_eff, p)
    churn_base = n_eff * (p.sigma * math.sqrt(2.0 / math.pi))
    W = (
        p.eta * (n_eff * p.N * M)
        - p.mu * M
        - p.lam * churn_base
        - 0.5 * p.rho * (n_eff * M * M)
    )
    return W


def welfare_NE(nG: int, nP: int, p: Params, profit_mode: str = "stable"):
    """
    Expected welfare at NE for given (nG,nP).
    """
    n = nG + nP
    if n <= 0:
        return float("nan")

    MG, MP = M_NE(n, p)
    cG, cP = per_provider_churn(nG, nP, p)
    piG, piP = per_provider_profit(nG, nP, p, profit_mode=profit_mode)

    # Aggregates
    Mbar = (nG * MG + nP * MP) / n
    churn = nG * cG + nP * cP
    Q = nG * (MG * MG) + nP * (MP * MP)
    profit_sum = nG * piG + nP * piP

    W = (
        p.eta * profit_sum
        - p.mu * Mbar
        - p.lam * churn
        - 0.5 * p.rho * Q
    )
    return W


def poa(nG: int, nP: int, p: Params, profit_mode: str = "stable"):
    Wso = welfare_SO(nG, nP, p)
    Wne = welfare_NE(nG, nP, p, profit_mode=profit_mode)
    if Wne == 0 or not np.isfinite(Wne):
        return float("nan")
    return Wso / Wne


# %% [markdown]
# Birth–Death step: death probs, sampling, update
# 
# Entry mode
# 
# entry_mode="stationary": stabilizza n(t) (ottimo per capire PoA e churn senza drift enorme).
# 
# entry_mode="exogenous": utile per stress test (mercato che cresce o collassa).

# %%
def death_probabilities(nG: int, nP: int, p: Params, profit_mode: str = "stable"):
    """
    p_t^death = delta_t * exp(-gamma_pi*pi_t + gamma_c*c_t)
    Then clipped into [0,1].
    """
    piG, piP = per_provider_profit(nG, nP, p, profit_mode=profit_mode)
    cG, cP = per_provider_churn(nG, nP, p)

    pDG = p.delta_G * math.exp(-p.gamma_pi * piG + p.gamma_c * cG)
    pDP = p.delta_P * math.exp(-p.gamma_pi * piP + p.gamma_c * cP)

    # clip to [0,1]
    pDG = min(max(pDG, 0.0), 1.0)
    pDP = min(max(pDP, 0.0), 1.0)
    return pDG, pDP


def step_birth_death(nG: int, nP: int, p: Params, rng: np.random.Generator,
                     profit_mode: str = "stable", entry_mode: str = "exogenous"):
    """
    One period transition (nG,nP) -> (nG_next, nP_next).
    entry_mode:
      - "exogenous": B ~ Poisson(lambda_B)
      - "stationary": lambda_B = E[D_G + D_P | state] (approx using n*p_death)
    """
    # Death
    pDG, pDP = death_probabilities(nG, nP, p, profit_mode=profit_mode)
    DG = rng.binomial(nG, pDG) if nG > 0 else 0
    DP = rng.binomial(nP, pDP) if nP > 0 else 0

    nG_surv = nG - DG
    nP_surv = nP - DP

    # Birth
    if entry_mode == "exogenous":
        lamB = p.lambda_B
    elif entry_mode == "stationary":
        # expected deaths approx:
        lamB = nG * pDG + nP * pDP
    else:
        raise ValueError("entry_mode must be 'exogenous' or 'stationary'")

    B = rng.poisson(lamB) if lamB > 0 else 0
    BG = rng.binomial(B, p.q_G) if B > 0 else 0
    BP = B - BG

    nG_next = nG_surv + BG
    nP_next = nP_surv + BP
    return nG_next, nP_next, {"DG": DG, "DP": DP, "B": B, "BG": BG, "BP": BP, "pDG": pDG, "pDP": pDP}


# %%
def simulate_one_run(nG0: int, nP0: int, p: Params, T: int,
                     seed: int = 0, profit_mode: str = "stable", entry_mode: str = "exogenous"):
    rng = np.random.default_rng(seed)

    nG = int(nG0)
    nP = int(nP0)

    # storage
    out = {
        "nG": np.zeros(T+1, dtype=int),
        "nP": np.zeros(T+1, dtype=int),
        "n": np.zeros(T+1, dtype=int),
        "M_G_NE": np.zeros(T+1),
        "M_P_NE": np.zeros(T+1),
        "M_SO": np.zeros(T+1),
        "churn": np.zeros(T+1),
        "poa": np.zeros(T+1),
        "W_SO": np.zeros(T+1),
        "W_NE": np.zeros(T+1),
    }

    def record(t, nG, nP):
        n = nG + nP
        MG, MP = M_NE(n, p) if n > 0 else (np.nan, np.nan)
        MSO = M_SO(n, p) if n > 0 else np.nan
        cG, cP = per_provider_churn(nG, nP, p) if n > 0 else (np.nan, np.nan)
        churn = nG*cG + nP*cP if n > 0 else np.nan
        Wso = welfare_SO(nG, nP, p) if n > 0 else np.nan
        Wne = welfare_NE(nG, nP, p, profit_mode=profit_mode) if n > 0 else np.nan
        P = (Wso / Wne) if (n > 0 and Wne != 0 and np.isfinite(Wne)) else np.nan

        out["nG"][t] = nG
        out["nP"][t] = nP
        out["n"][t] = n
        out["M_G_NE"][t] = MG
        out["M_P_NE"][t] = MP
        out["M_SO"][t] = MSO
        out["churn"][t] = churn
        out["poa"][t] = P
        out["W_SO"][t] = Wso
        out["W_NE"][t] = Wne

    record(0, nG, nP)

    for t in range(T):
        nG, nP, _info = step_birth_death(nG, nP, p, rng, profit_mode=profit_mode, entry_mode=entry_mode)
        record(t+1, nG, nP)

    return out


def simulate_monte_carlo(nG0: int, nP0: int, p: Params, T: int, R: int,
                         seed: int = 0, profit_mode: str = "stable", entry_mode: str = "exogenous"):
    runs = []
    for r in range(R):
        runs.append(simulate_one_run(nG0, nP0, p, T, seed=seed+r, profit_mode=profit_mode, entry_mode=entry_mode))
    return runs


def summarize_runs(runs, key: str):
    """
    Stack a time series across runs and return mean + quantiles.
    """
    X = np.vstack([run[key] for run in runs])  # shape (R, T+1)
    mean = X.mean(axis=0)
    q10 = np.quantile(X, 0.10, axis=0)
    q50 = np.quantile(X, 0.50, axis=0)
    q90 = np.quantile(X, 0.90, axis=0)
    return mean, q10, q50, q90


# %%
def plot_with_bands(t, mean, q10, q90, title, ylabel):
    plt.figure()
    plt.plot(t, mean)
    plt.fill_between(t, q10, q90, alpha=0.2)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.show()



# %% [markdown]
# Cosa misura e cosa aspettarsi
# 
# 𝑀^{\bar}(t): margine medio nel mercato. Con più concorrenza (n alto) tende a stare relativamente basso (in questo modello scende con 
# logn
# 
# share_G(t): quota di grandi tra i provider; può driftare se i piccoli muoiono più spesso (𝛿𝑃>𝛿𝐺).
# 
# var_M(t): dispersione dei margini (eterogeneità). Con 
# 𝐾𝐺≠𝐾𝑃 non è zero e cresce se la composizione diventa “mixata”.

# %%
def add_derived_metrics(run):
    """
    Adds derived time series to a run dict:
    - M_bar(t): avg markup across providers
    - share_G(t): nG/n
    - var_M(t): cross-sectional variance of markups across providers (two-point mass)
    """
    n = run["n"].astype(float)
    nG = run["nG"].astype(float)
    nP = run["nP"].astype(float)
    MG = run["M_G_NE"]
    MP = run["M_P_NE"]

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        shareG = np.where(n > 0, nG / n, np.nan)
        Mbar = np.where(n > 0, (nG * MG + nP * MP) / n, np.nan)

    # Two-type variance: Var = w_G (MG - Mbar)^2 + w_P (MP - Mbar)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        wG = np.where(n > 0, nG / n, np.nan)
        wP = np.where(n > 0, nP / n, np.nan)
        varM = wG * (MG - Mbar) ** 2 + wP * (MP - Mbar) ** 2

    run["share_G"] = shareG
    run["M_bar"] = Mbar
    run["var_M"] = varM
    return run


def add_metrics_to_all_runs(runs):
    return [add_derived_metrics(run) for run in runs]


# %% [markdown]
# helper plot (mean + bande)

# %%
def plot_with_bands(t, mean, q10, q90, title, ylabel):
    plt.figure()
    plt.plot(t, mean)
    plt.fill_between(t, q10, q90, alpha=0.2)
    plt.title(title)
    plt.xlabel("t (months)")
    plt.ylabel(ylabel)
    plt.show()


# %% [markdown]
# helper plot (mean + bande)

# %%
def plot_timeseries_bands(runs, T, keys_and_labels):
    t = np.arange(T+1)
    for key, title, ylabel in keys_and_labels:
        mean, q10, q50, q90 = summarize_runs(runs, key)
        plot_with_bands(t, mean, q10, q90, title, ylabel)


# %% [markdown]
# grafici “must-have” time series
# 
# 
# n(t) con entry_mode="stationary": tende a oscillare attorno al livello iniziale, con rumore (run-to-run).
# 
# n_G(t), n_P(t): se piccoli più fragili, spesso vedrai n_P scendere e share_G crescere nel tempo (dipende da 
# 𝛿𝑃,𝛾𝜋).
# 
# PoA(t): può aumentare quando churn/dispersione cresce, o diminuire se prezzi medi migliorano. La cosa interessante è vedere regimi: fasi stabili vs fasi turbolente.
# 
# churn C(t): cresce con mix 
# 𝑛𝐺𝑛𝑃 e con log𝑛 nel tuo modello; col stationary entry ci aspetti fluttuazioni attorno a una banda.
# 
# var_M(t): se share_G tende a 0 o 1, varianza può calare (mercato quasi tutto di un tipo); se share_G resta ~0.5, varianza alta.

# %%
# Ensure derived metrics exist
#runs = add_metrics_to_all_runs(runs)

#T = len(runs[0]["n"]) - 1

keys_and_labels = [
    ("n", "Total providers over time", "n(t)"),
    ("nG", "Large providers over time", "n_G(t)"),
    ("nP", "Small providers over time", "n_P(t)"),
    ("share_G", "Share of large providers over time", "n_G(t)/n(t)"),
    ("poa", "Price of Anarchy over time", "PoA(t)"),
    ("churn", "Expected churn over time", "C(t)"),
    ("M_bar", "Average Nash markup over time", "M̄^NE(t)"),
    ("var_M", "Markup dispersion over time", "Var(M^NE)(t)"),
    ("M_G_NE", "Nash markup (large providers)", "M_G^NE(t)"),
    ("M_P_NE", "Nash markup (small providers)", "M_P^NE(t)"),
    ("M_SO", "Social Optimum markup", "M^SO(t)"),
]

#plot_timeseries_bands(runs, T, keys_and_labels)


# %% [markdown]
# istribuzioni “in regime” (ultimi mesi): istogrammi / densità
# 
# Per evitare storytelling, in paper si mostra spesso la distribuzione in una finestra “stationary”.

# %%
def collect_stationary_samples(runs, key, burn_in=60, last_window=60):
    """
    Collects samples from the last_window months, after burn_in.
    Returns a 1D array pooling over runs and time.
    """
    samples = []
    for run in runs:
        T = len(run[key]) - 1
        start = max(burn_in, T - last_window)
        x = run[key][start:T+1]
        x = x[np.isfinite(x)]
        samples.append(x)
    if len(samples) == 0:
        return np.array([])
    return np.concatenate(samples)


# %%
def plot_hist(samples, title, xlabel, bins=40):
    plt.figure()
    plt.hist(samples, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.show()


# %% [markdown]
# Cosa aspettarsi
# 
# PoA: spesso non è “una campana”, ma una distribuzione con code (periodi turbolenti).
# 
# n: se stationary entry funziona, distribuzione centrata attorno a ~700.
# 
# share_G: se piccoli muoiono più spesso, puoi vedere massa spostata verso share_G più alta.

# %%
burn_in = 60        # 5 years if monthly
last_window = 60    # last 5 years
"""
for key, title, xlabel in [
    ("poa", "Stationary distribution of PoA", "PoA"),
    ("n", "Stationary distribution of total providers", "n"),
    ("churn", "Stationary distribution of churn", "C"),
    ("M_bar", "Stationary distribution of average markup", "M̄^NE"),
    ("share_G", "Stationary distribution of share of large providers", "n_G/n"),
]:
    samples = collect_stationary_samples(runs, key, burn_in=burn_in, last_window=last_window)
    #plot_hist(samples, title, xlabel, bins=40)

"""
# %% [markdown]
# Phase plot 
# (𝑛𝐺,𝑛𝑃): traiettorie nel piano
# 
# Questo è molto utile per mostrare come la dinamica “vaga” nello spazio degli stati.
# 
# 
# osa aspettarsi
# 
# con 𝛿𝑃>𝛿𝐺, le traiettorie possono “migrare” verso zone con meno piccoli.
# 
# se entry è stationary e q_G costante, dovrebbe esserci una nube “stazionaria”.

# %%

#plot_phase_trajectories(runs, n_traj=30, burn_in=0)


# %% [markdown]
# Heatmap: PoA in funzione di 
# (𝑛𝐺,𝑛𝑃)
# Questo è uno dei grafici più potenti: mostra dove nello spazio stati la PoA è alta/bassa.
# 
# Qui non serve simulare: calcoliamo 
# PoA(𝑛𝐺,𝑛𝑃) direttamente dalla formula.

# %%
def poa_grid(p: Params, n_min=200, n_max=900, n_step=10, share_grid=None, profit_mode="stable"):
    """
    Compute PoA on a grid of (n, shareG). Returns mesh arrays and PoA values.
    """
    if share_grid is None:
        share_grid = np.linspace(0.05, 0.95, 19)  # avoid extremes 0 or 1

    n_vals = np.arange(n_min, n_max + 1, n_step)
    S = np.array(share_grid)

    PoA = np.zeros((len(S), len(n_vals)))

    for i, s in enumerate(S):
        for j, n in enumerate(n_vals):
            nG = int(round(s * n))
            nP = n - nG
            if n < 2 or nG < 1 or nP < 1:
                PoA[i, j] = np.nan
                continue
            PoA[i, j] = poa(nG, nP, p, profit_mode=profit_mode)

    return n_vals, S, PoA


def plot_heatmap(n_vals, share_vals, Z, title, xlabel="n", ylabel="share_G"):
    plt.figure()
    # extent: [xmin, xmax, ymin, ymax]
    extent = [n_vals.min(), n_vals.max(), share_vals.min(), share_vals.max()]
    plt.imshow(Z, aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label="PoA")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Example heatmap
p = Params()
n_vals, share_vals, Z = poa_grid(p, n_min=200, n_max=900, n_step=10, profit_mode="stable")
#plot_heatmap(n_vals, share_vals, Z, "PoA heatmap as function of (n, share_G)")


# %%
def poa_safe(nG: int, nP: int, p: Params, profit_mode="stable", eps=1e-9):
    Wso = welfare_SO(nG, nP, p)
    Wne = welfare_NE(nG, nP, p, profit_mode=profit_mode)

    # Mask ill-defined / misleading regions
    if not np.isfinite(Wso) or not np.isfinite(Wne):
        return np.nan
    if Wso <= 0:
        return np.nan
    if Wne <= eps:
        return np.nan

    return Wso / Wne


def diagnose_outliers(p: Params, n_min=200, n_max=900, n_step=10, share_grid=None, profit_mode="stable"):
    if share_grid is None:
        share_grid = np.linspace(0.05, 0.95, 19)

    n_vals = np.arange(n_min, n_max + 1, n_step)
    S = np.array(share_grid)

    rows = []
    for s in S:
        for n in n_vals:
            nG = int(round(s*n))
            nP = n - nG
            if n < 2 or nG < 1 or nP < 1:
                continue
            Wso = welfare_SO(nG, nP, p)
            Wne = welfare_NE(nG, nP, p, profit_mode=profit_mode)
            P = np.nan
            if np.isfinite(Wso) and np.isfinite(Wne) and Wne != 0:
                P = Wso / Wne
            rows.append((n, s, nG, nP, Wso, Wne, P))

    # show the worst denominators (closest to zero)
    rows_sorted = sorted(rows, key=lambda x: abs(x[5]))[:10]
    return rows_sorted

# Example: inspect problematic cells
bad = diagnose_outliers(Params(), profit_mode="stable")
#for r in bad:
    #print("n, shareG, nG, nP, Wso, Wne, PoA =", r)


# %%
def poa_grid_safe(p: Params, n_min=200, n_max=900, n_step=10, share_grid=None, profit_mode="stable"):
    if share_grid is None:
        share_grid = np.linspace(0.05, 0.95, 19)

    n_vals = np.arange(n_min, n_max + 1, n_step)
    S = np.array(share_grid)
    Z = np.full((len(S), len(n_vals)), np.nan)

    for i, s in enumerate(S):
        for j, n in enumerate(n_vals):
            nG = int(round(s * n))
            nP = n - nG
            if n < 2 or nG < 1 or nP < 1:
                continue
            Z[i, j] = poa_safe(nG, nP, p, profit_mode=profit_mode)

    return n_vals, S, Z


def plot_heatmap_clipped(n_vals, share_vals, Z, title, clip=(0.05, 0.95), xlabel="n", ylabel="share_G"):
    Zc = Z.copy()
    finite = Zc[np.isfinite(Zc)]
    if finite.size > 0:
        lo = np.quantile(finite, clip[0])
        hi = np.quantile(finite, clip[1])
        Zc = np.clip(Zc, lo, hi)

    plt.figure()
    extent = [n_vals.min(), n_vals.max(), share_vals.min(), share_vals.max()]
    plt.imshow(Zc, aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label=f"PoA (clipped to {int(100*clip[0])}-{int(100*clip[1])} pct)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


p = Params()
#n_vals, share_vals, Z = poa_grid_safe(p, n_min=200, n_max=900, n_step=10, profit_mode="stable")
#plot_heatmap_clipped(n_vals, share_vals, Z, "PoA heatmap (masked + clipped)", clip=(0.05, 0.95))


# %% [markdown]
# Fix #3 (più “da paper”): invece della PoA, plotti l’inefficienza in differenza ΔW
# 
# Molto spesso è più stabile plottare:
# 
# Δ𝑊(𝑛𝐺,𝑛𝑃)=𝐸[𝑊𝑆𝑂]−𝐸[𝑊𝑁𝐸]
# 
# non esplode
# 
# ha interpretazione immediata: “loss in welfare units”
# 
# ti evita il problema del denominatore
# 
# 
# Cosa aspettarti: di solito 
# Δ
# 𝑊
# ΔW è massimo quando:
# 
# churn e dispersione margini sono alti (mix alto, 
# Δ
# 𝐾
# ΔK alto)
# 
# 𝜆
# λ è alto (penalizzi churn)

# %%
#print(asdict(p))


# %% [markdown]
# Survival analysis: “lifetime” dei provider (proxy)
# 
# Qui non hai individui tracciati, ma puoi stimare proxy di lifetime via hazard media.
# Oppure (più realistico) tracci individui — ma è più complesso. Per ora ti do la proxy “da paper”.
# 
# 
# se piccoli hanno 
# 𝛿𝑃 maggiore, lifetime_P più corto.
# 
# se churn cresce, lifetimes possono diminuire.

# %%
def expected_lifetime_months(p_death: float):
    if p_death <= 0:
        return np.inf
    return 1.0 / p_death

def plot_expected_lifetimes_over_time(runs, p: Params, profit_mode="stable"):
    """
    For each time t, compute mean expected lifetimes using state-dependent death probabilities.
    """
    T = len(runs[0]["n"]) - 1
    lifG = []
    lifP = []

    # Use mean state across runs at each t for a clean plot
    mean_nG, _, _, _ = summarize_runs(runs, "nG")
    mean_nP, _, _, _ = summarize_runs(runs, "nP")

    for t in range(T+1):
        nG = int(round(mean_nG[t]))
        nP = int(round(mean_nP[t]))
        pDG, pDP = death_probabilities(nG, nP, p, profit_mode=profit_mode)
        lifG.append(expected_lifetime_months(pDG))
        lifP.append(expected_lifetime_months(pDP))

    tgrid = np.arange(T+1)
    plt.figure()
    plt.plot(tgrid, lifG, label="Large (expected months)")
    plt.plot(tgrid, lifP, label="Small (expected months)")
    plt.title("Proxy expected provider lifetime over time")
    plt.xlabel("t (months)")
    plt.ylabel("Expected lifetime (months)")
    plt.legend()
    plt.show()

#plot_expected_lifetimes_over_time(runs, p, profit_mode="stable")


# %% [markdown]
# Collapse risk”: probabilità che il mercato scenda sotto una soglia
# 
# Definisci “collasso” come 
# 𝑛(𝑡)<𝑛min
#  (es. 500).
# 
# con entry_mode="stationary" la probabilità dovrebbe restare bassa (se parametri non estremi).
# 
# con entry exogenous basso o death alto, può crescere nel tempo.

# %%
def collapse_probability_over_time(runs, n_min=500):
    X = np.vstack([run["n"] for run in runs])  # (R, T+1)
    prob = (X < n_min).mean(axis=0)
    return prob

def plot_collapse_probability(runs, n_min=500):
    prob = collapse_probability_over_time(runs, n_min=n_min)
    t = np.arange(len(prob))
    plt.figure()
    plt.plot(t, prob)
    plt.title(f"Probability of 'market collapse': P(n(t) < {n_min})")
    plt.xlabel("t (months)")
    plt.ylabel("probability")
    plt.show()

#plot_collapse_probability(runs, n_min=500)


# %% [markdown]
# on entry_mode="stationary" la probabilità dovrebbe restare bassa (se parametri non estremi).
# 
# con entry exogenous basso o death alto, può crescere nel tempo.

# %%
def stationary_summary(runs, key, burn_in=60, last_window=60):
    samples = collect_stationary_samples(runs, key, burn_in=burn_in, last_window=last_window)
    if samples.size == 0:
        return {"mean": np.nan, "q10": np.nan, "q50": np.nan, "q90": np.nan}
    return {
        "mean": float(samples.mean()),
        "q10": float(np.quantile(samples, 0.10)),
        "q50": float(np.quantile(samples, 0.50)),
        "q90": float(np.quantile(samples, 0.90)),
    }


# %%
def sweep_param_and_plot(base_params: Params, nG0: int, nP0: int, T: int, R: int,
                         sweep_name: str, sweep_values, profit_mode="stable", entry_mode="stationary",
                         burn_in=60, last_window=60, seed=0):
    means = []
    q10s = []
    q90s = []

    for v in sweep_values:
        p = Params(**asdict(base_params))
        setattr(p, sweep_name, float(v))

        runs = simulate_monte_carlo(nG0, nP0, p, T=T, R=R, seed=seed,
                                    profit_mode=profit_mode, entry_mode=entry_mode)
        runs = add_metrics_to_all_runs(runs)

        summ = stationary_summary(runs, "poa", burn_in=burn_in, last_window=last_window)
        means.append(summ["mean"])
        q10s.append(summ["q10"])
        q90s.append(summ["q90"])

    x = np.array(sweep_values, dtype=float)
    means = np.array(means)
    q10s = np.array(q10s)
    q90s = np.array(q90s)

    plt.figure()
    plt.plot(x, means)
    plt.fill_between(x, q10s, q90s, alpha=0.2)
    plt.title(f"Stationary PoA vs {sweep_name}")
    plt.xlabel(sweep_name)
    plt.ylabel("PoA (stationary)")
    plt.show()


# %% [markdown]
# Cosa aspettarsi
# 
# aumentando λ, PoA tende a peggiorare (più peso al churn).
# 
# aumentando 
# Δ𝐾 (via 𝐾𝐺), aumenta dispersione e switching prezzo-driven → PoA spesso peggiora.

# %%

def main():
    base_p = Params()
    T = 120
    R = 150
    n0 = 700
    nG0 = int(0.2 * n0)
    nP0 = n0 - nG0

    # Sweep λ (churn penalty)
    sweep_param_and_plot(base_p, nG0, nP0, T, R, "lam", [0.0, 0.5, 1.0, 1.5, 2.0],
                        profit_mode="stable", entry_mode="stationary")

    # Sweep K_G (thus ΔK changes)
    sweep_param_and_plot(base_p, nG0, nP0, T, R, "K_G", [0.02, 0.05, 0.10, 0.15, 0.20],
                        profit_mode="stable", entry_mode="stationary")

    p = Params()
    T = 600
    R = 200

    n0 = 700
    nG0 = int(0.2 * n0)
    nP0 = n0 - nG0

    runs = simulate_monte_carlo(nG0, nP0, p, T=T, R=R, seed=42,
                                profit_mode="stable", entry_mode="stationary")
    runs = add_metrics_to_all_runs(runs)

    # Time series with bands
    plot_timeseries_bands(runs, T, keys_and_labels)

    # Stationary histograms
    burn_in = 60
    last_window = 60
    for key, title, xlabel in [
        ("poa", "Stationary distribution of PoA", "PoA"),
        ("n", "Stationary distribution of total providers", "n"),
        ("churn", "Stationary distribution of churn", "C"),
        ("M_bar", "Stationary distribution of average markup", "M̄^NE"),
        ("share_G", "Stationary distribution of share of large providers", "n_G/n"),
    ]:
        samples = collect_stationary_samples(runs, key, burn_in=burn_in, last_window=last_window)
        plot_hist(samples, title, xlabel, bins=40)

    # Phase plot
    #plot_phase_trajectories(runs, n_traj=30, burn_in=0)

    # Heatmap PoA(n, shareG)
    n_vals, share_vals, Z = poa_grid(p, n_min=200, n_max=900, n_step=10, profit_mode="stable")
    plot_heatmap(n_vals, share_vals, Z, "PoA heatmap as function of (n, share_G)")

    # Lifetime proxy
    plot_expected_lifetimes_over_time(runs, p, profit_mode="stable")

    # Collapse risk
    plot_collapse_probability(runs, n_min=500)

    p = Params()
    T = 120           # es. 120 mesi = 10 anni se 1 step = 1 mese
    R = 200

    # esempio iniziale: n=700 con 20% grandi
    n0 = 700
    nG0 = int(0.2 * n0)
    nP0 = n0 - nG0

    runs = simulate_monte_carlo(nG0, nP0, p, T=T, R=R, seed=42,
                                profit_mode="stable", entry_mode="stationary")

    t = np.arange(T+1)

    for key, title, ylabel in [
        ("n", "Total providers over time", "n(t)"),
        ("poa", "Price of Anarchy over time", "PoA(t)"),
        ("churn", "Expected churn over time", "C(t)"),
        ("M_G_NE", "NE markup (large providers)", "M_G^NE"),
        ("M_P_NE", "NE markup (small providers)", "M_P^NE"),
    ]:
        mean, q10, q50, q90 = summarize_runs(runs, key)
        plot_with_bands(t, mean, q10, q90, title, ylabel)


if __name__ == "__main__":
    main()


# %%
def run_mc_experiment(n0, share_G, p, T=600, R=200, seed=0):
    """
    Runs a Monte Carlo experiment and returns aggregated observables
    suitable for ML.
    """

    nG0 = int(share_G * n0)
    nP0 = n0 - nG0

    runs = simulate_monte_carlo(
        nG0, nP0, p,
        T=T, R=R,
        seed=seed,
        profit_mode="stable",
        entry_mode="stationary"
    )

    runs = add_metrics_to_all_runs(runs)

    return runs


# %%
def extract_observables(runs, burn_in=60, last_window=60):

    obs = {}

    for key in ["poa", "M_bar", "n", "churn"]:
        samples = collect_stationary_samples(
            runs,
            key,
            burn_in=burn_in,
            last_window=last_window
        )

        obs[f"{key}_mean"] = np.mean(samples)
        obs[f"{key}_std"]  = np.std(samples)
        obs[f"{key}_amp"]  = np.percentile(samples, 90) - np.percentile(samples, 10)

    return obs


def temporal_stats(x):
    return {
        "mean": np.mean(x),
        "std":  np.std(x),
        "p10":  np.percentile(x, 10),
        "p90":  np.percentile(x, 90),
    }

def convergence_time(x, tol=0.05):
    x = np.asarray(x)
    x_final = np.mean(x[int(0.8 * len(x)):])
    band = tol * abs(x_final)

    for t in range(len(x)):
        if np.all(np.abs(x[t:] - x_final) < band):
            return t
    return len(x)


# %%



