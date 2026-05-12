from montecarlo_simulation import Params, run_mc_experiment, extract_observables
import numpy as np
import pandas as pd


N_SAMPLES = 50000

# Parameter ranges (coerenti con MC)
THETA_COST_RANGE = (0.5, 2.0)
THETA_CONG_RANGE = (0.1, 1.0)

# Measurement noise
SIGMA_FLOW = 0.02
SIGMA_CONG = 0.01

FEATURE_NAMES = [
    "poa_mean", "poa_std", "poa_amp",
    "M_bar_mean", "M_bar_std", "M_bar_amp",
    "n_mean", "n_std", "n_amp",
    "churn_mean", "churn_std", "churn_amp"
]


def sample_demand(Q0=100.0, beta=0.05, m_bar=1.0):
    noise = np.random.normal(0, 2.0)
    return Q0 * np.exp(-beta * m_bar) + noise


def generate_sample():
    p = Params()
    # 1. Sample latent parameters (TARGET ML)
    theta_cost = np.random.uniform(0.5, 2.0)
    theta_cong = np.random.uniform(0.1, 1.0)

    # 2. Inject them into Params
    p.theta_cost = theta_cost
    p.theta_cong = theta_cong

    # 3. Sample initial conditions
    n0 = np.random.randint(500, 900)
    share_G = np.random.uniform(0.1, 0.5)

    # 4. Run MC
    runs = run_mc_experiment(n0, share_G, p, T=100, R = 30)

    # 5. Extract observables
    obs = extract_observables(runs)

    # 6. Feature vector
    X = np.array([obs[name] for name in FEATURE_NAMES])

    y = np.array([theta_cost, theta_cong])

    feature_names = list(obs.keys())

    return X, y



def generate_dataset(N=1000, seed=123):
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []

    for i in range(N):
        X, y = generate_sample()
        X_list.append(X)
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    if (i+1) % 20 == 0:
        print(f"Generated {i+1}/{N}")

    return X, y



def save_dataset(X, y, filename="mc_ml_dataset.npz"):
    
    np.savez(
    filename,
    X=X,
    y=y,
    feature_names=np.array(FEATURE_NAMES),
    target_names=np.array(["theta_cost", "theta_cong"])
    )

    return X, y


