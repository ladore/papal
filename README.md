# ResonanceBayesianTracker – Narrative Bayesian Evidence Ledger

A lightweight, single-file Python framework for tracking and updating multiple competing narrative hypotheses using Bayesian inference, market signals, narrative geometry (PCA), and a discrete chaos/stability diagnostic.

Built for scenarios where analysts need an **inspectable, disciplined trail** of priors, evidence likelihoods, market data, and posterior evolution – without committing to a full causal model.

---

## Why This Exists

Narrative‑driven domains (policy, geopolitics, technology adoption) are plagued by **confirmation bias** and **hand‑waving**. This tracker forces you to:

- Make your priors explicit.  
- Record every evidence event with a *likelihood ratio* for each hypothesis.  
- Separate *market signals* (price + liquidity) from *qualitative evidence*.  
- Watch how posteriors evolve over time.  
- Diagnose whether the posterior trajectory is **stable, diverging, or saddle‑like** using a simple Lyapunov‑style proxy.  
- Visualise the latent structure of your narrative dimensions with PCA.

It does **not** prove causality. It **does** make your reasoning auditable.

---

## Features

- **Multiple hypotheses** – Pre‑loaded set (`governance_influence`, `symbolic_legitimation`, `anthropic_brand_benefit`, `low_policy_impact`) or fully custom.  
- **Evidence ledger** – Add events with per‑hypothesis likelihood ratios, confidence‑weighted tempering, and automatic Bayesian updates.  
- **Market signals** – Register prediction‑market prices with liquidity‑weighted signal extraction.  
- **Posterior history** – Every update is timestamped and logged for timeline visualisation.  
- **Narrative PCA** – Principal component analysis over a 7‑dimensional narrative feature space (optional; requires `scikit‑learn`).  
- **Chaos / Stability diagnostic** – Treats the posterior vector as a discrete dynamical system; computes step norms, local growth rates, and a Lyapunov‑style proxy to classify the trajectory.  
- **Exportable reports** – Generate a full Markdown report containing posteriors, evidence tables, market signals, PCA, and stability analysis.  
- **Plotting** – Matplotlib timeline of posterior probabilities (optional; requires `matplotlib`).  

---

## Quick Start

### Install

```bash
# Core (always works)
pip install numpy scikit-learn matplotlib  # optional but recommended
```

The tracker runs with **no external dependencies** – only Python standard library. The optional packages add PCA, plotting, and scaling.

### Run the demo

```python
from resonance_tracker import build_demo_tracker, ResonanceBayesianTracker

tracker = build_demo_tracker()          # pre‑loaded scenario
tracker.show_polymarket_summary()
tracker.run_chaos_stability_analysis()
pca = tracker.run_narrative_pca()
tracker.plot_probability_timeline()
```

### Minimal custom usage

```python
tracker = ResonanceBayesianTracker()

tracker.add_event(
    EvidenceEvent.from_date(
        date_str="2026-06-01",
        title="New regulatory proposal",
        quality="primary",
        source="official gazette",
        rationale="Direct policy signal.",
        confidence=0.9,
        likelihood_ratios={
            "governance_influence": 2.0,
            "symbolic_legitimation": 1.2,
            "anthropic_brand_benefit": 1.1,
            "low_policy_impact": 0.6,
        },
    )
)

tracker.add_polymarket_bet(
    "2026-06-01",
    "Will regulation pass before 2027?",
    0.55,
    "250k",
    relevance={"governance_influence": 0.8},
)

print(tracker.posterior_table())
```

---

## Understanding the Output

### Posteriors

Each hypothesis gets a probability updated via Bayes’ rule. The legend explains plain‑language mappings:

| Posterior range | Interpretation |
|----------------|----------------|
| 0–5%           | effectively ruled out |
| 5–20%          | unlikely |
| 20–40%         | possible but not favoured |
| 40–60%         | uncertain / roughly balanced |
| 60–75%         | more likely than not |
| 75–90%         | strongly favoured |
| 90–97%         | very strongly favoured |
| 97–100%        | near certain within this model |

### Market signals

A liquidity‑weighted signal is computed: `(yes_odds - 0.5) × liquidity_weight`. Weights are based on volume thresholds to prevent thin markets from driving conclusions.

### Chaos / Stability diagnostic

The posterior vector over hypotheses is treated as a **discrete dynamical system**. Each event‑to‑event movement is a step; the diagnostic measures:

- **Step norms** – Euclidean length of the posterior delta vector.  
- **Growth ratios** – How step norms change over time.  
- **Lyapunov‑style proxy** – Average log‑ratio of consecutive step norms. Negative → shrinking (stable); positive → growing (unstable).  
- **Classification** – One of: `stable/converging`, `weakly stable`, `unstable/diverging`, `saddle‑like mixed stability`, `indeterminate / near transition`.

This is a **local stability read**, not a rigorous chaos proof. It is meant to flag whether your belief trajectory is settling or fluttering.

### Narrative PCA

A static analysis maps narrative items (e.g. events, themes) onto two principal components:

- **Moral Legitimacy Axis** – how strongly an item invokes moral framing.  
- **Policy / Market Action Axis** – how strongly it links to policy or market action.

Loadings show which narrative dimensions drive each axis.

---

## Dependencies

| Package | Required? | Purpose |
|---------|-----------|---------|
| Python ≥3.9 | yes | Core |
| `numpy` | yes | PCA & scaling (optional) |
| `scikit-learn` | yes | PCA & scaling (optional) |
| `matplotlib` | yes | Timeline plots (optional) |

If a dependency is missing, the relevant methods print a message and skip.

---

## Project Structure

All code is in **a single file** – `resonance_tracker.py` (or whatever you name it). Classes and helper functions are self‑contained; no imports beyond the standard library for the core logic.

```
resonance_tracker.py
├── ResonanceBayesianTracker      # main class
├── Hypothesis                     # dataclass for each narrative hypothesis
├── EvidenceEvent / MarketSignal   # evidence & market signal containers
├── PosteriorPoint                 # logged posterior update
├── NarrativePCAResult / ChaosStabilityResult  # result containers
└── helper functions               # formatting, probability phrases, stability classification
```

---

## Configuration

You can provide your own hypotheses when instantiating:

```python
custom_hypotheses = [
    Hypothesis(key="regulatory_pass", label="Regulation passes before 2027", prior=0.3, description="..."),
    Hypothesis(key="industry_capture", label="Industry captures regulatory process", prior=0.5, ...),
]
tracker = ResonanceBayesianTracker(hypotheses=custom_hypotheses)
```

All evidence events accept per‑hypothesis likelihood ratios. Confidence (0–1) tempers the ratio via: `LR_weighted = LR ^ confidence`.

---

## Exporting Reports

```python
tracker.export_report("report.md", pca_result=pca, chaos_result=chaos)
```

Produces a self‑contained Markdown file with all tables, legends, and diagnostic explanations.

---

## License

MIT – use freely, adapt, and share. The author provides no warranty; this is a reasoning tool, not a financial or policy advisor.

---

## Contribute

Issues, forks, and pull requests welcome. The tracker is intentionally simple – improvements that preserve **inspectability** and **single‑file readability** are preferred.

---

## Theoretical Caveats

- Bayesian updates assume **conditional independence** of evidence events – a known simplification.  
- The chaos diagnostic uses **step norms in posterior space**, not a true Lyapunov exponent on a smooth dynamical system. It is a heuristic proxy for local stability.  
- PCA is performed on a **fixed set of narrative dimensions** and a **hand‑crafted data matrix** – it reflects the current scenario, not a general‑purpose semantic space.  

Use the tool to **discipline your reasoning**, not to **automate your beliefs**.

