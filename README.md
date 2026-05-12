# Game Theory of Infrastructure Markets

A game-theoretic and computational study of competition in infrastructure-based markets, with a focus on gas transportation networks in which multiple providers strategically set access prices.

The project develops an intermediate model between classical Cournot and Bertrand competition, studies decentralized market equilibria, compares them with a social optimum, and quantifies market inefficiency through the **Price of Anarchy**. The analytical model is complemented by numerical simulations and machine-learning tools for exploring the behavior of the system across a broad parameter space.

---

## Overview

This work studies a gas transportation market in which several infrastructure providers compete for clients by choosing access margins. The motivation comes from the rapid expansion of the gas sector in Friuli-Venezia-Giulia, where the number of active firms increased significantly over a short period of time.

The market cannot be described purely as:

- **Cournot competition**, because firms do not only compete through quantities;
- **Bertrand competition**, because infrastructure constraints, client switching, and imperfect substitutability between providers matter.

Instead, the project proposes a hybrid framework in which providers choose price margins, clients react to relative pricing, and the resulting decentralized equilibrium is compared with a socially optimal allocation.

---

## Market Model

The market consists of `n` providers. Each provider `i` chooses a margin:

$$
M_i
$$

Provider `i` has an initial client base of value:

$$
N_i
$$

Competition induces a variation in its client base:

$$
\Delta_i = F(\bar{M}_{-i} - M_i)
$$

where:

$$
\bar{M}_{-i} = \frac{1}{n-1}\sum_{j \neq i} M_j
$$

If provider `i` sets a lower margin than its competitors, it attracts clients. If it sets a higher margin, it loses clients.

The provider utility is modeled as:

$$
u_i(M_i, M_{-i}) = M_i(N_i + \Delta_i) + k\Delta_i
$$

where the first term represents revenue from the client base and the second term captures the effect of client variation.

---

## Nash Equilibrium and Social Optimum

The project compares two market outcomes:

### Nash Equilibrium

Each provider independently chooses its own margin to maximize its private utility.

This represents the decentralized competitive outcome.

### Social Optimum

A social welfare function is introduced to evaluate the globally efficient allocation. The welfare function accounts for:

- provider revenues,
- consumer/client-side effects,
- switching costs or churn,
- inefficiencies due to price dispersion,
- production or infrastructure-related costs.

---

## Price of Anarchy

Market inefficiency is quantified using the **Price of Anarchy**:

$$
PoA = \frac{W_{SO}}{W_{NE}}
$$

where:

- $W_{SO}$ is the welfare at the social optimum;
- $W_{NE}$ is the welfare at the Nash equilibrium.

A higher Price of Anarchy indicates a larger welfare loss caused by decentralized strategic behavior.

---

## Heterogeneous Providers

The model is extended to include heterogeneous firms, distinguishing between:

- **large providers**;
- **small providers**.

This allows the analysis of asymmetric market structures, where providers differ in stability, client base, growth incentives, and probability of market exit.

The simulations track how the composition of the market evolves over time and how this affects:

- average Nash margins;
- total number of providers;
- provider churn;
- welfare;
- Price of Anarchy;
- survival of small and large firms.

---

## Numerical Simulations

The project includes Monte Carlo simulations of provider dynamics over time.

The simulation framework models:

- provider entry;
- provider exit;
- birth–death dynamics;
- client churn;
- Nash equilibrium margins;
- social optimum margins;
- welfare at equilibrium;
- welfare at the social optimum;
- Price of Anarchy over time.

The main simulation logic is implemented in:

```text
montecarlo_simulation.py
