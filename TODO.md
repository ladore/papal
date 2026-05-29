# Cumulative Phase: Full System Upgrade

## 1. Dynamical Systems – True Lyapunov Exponents & Attractor Diagnostics

**Current state**  
- Takens embedding is implemented.  
- Recurrence quantification (rate, determinism, laminarity) is computed.  
- Lyapunov-style proxy uses simple log-ratios of step norms.

**What remains** (detailed)

### 1.1 Maximal Lyapunov Exponent (MLE) via Rosenstein Algorithm
Replace the heuristic proxy with the standard Rosenstein algorithm (or Kantz algorithm) applied to the embedded state vectors.

- **Implementation**:  
  - For each embedded point, find its nearest neighbour in phase space (avoiding temporal neighbours).  
  - Track the average divergence over successive time steps.  
  - The slope of the log-divergence curve gives the MLE.  
- **Rationale**: A true MLE is a rigorous dynamical invariant. A positive MLE indicates chaos; negative indicates stability; zero indicates a periodic/limit cycle.  
- **Effort**: Medium. Requires careful nearest‑neighbour search and averaging over multiple trajectories.

### 1.2 Correlation Dimension (D2)
Estimate the fractal dimension of the attractor from the correlation integral.

- **Implementation**:  
  - Compute the correlation integral `C(r)` for a range of radii `r`.  
  - The slope of `log C(r)` vs `log r` in the scaling region gives D2.  
- **Rationale**: D2 distinguishes low‑dimensional (deterministic) from high‑dimensional (noise‑dominated) dynamics. For a belief trajectory, D2 close to 1–3 suggests a low‑dimensional attractor.  
- **Effort**: Medium. Standard implementation, but needs careful selection of scaling region.

### 1.3 Recurrence Plot (RP) Export & Visualization
Export the binary recurrence matrix and provide a Matplotlib `imshow` plot.

- **Implementation**:  
  - The recurrence matrix is already computed; add a method to plot it.  
  - Colourmap: black for recurrent, white for non‑recurrent.  
- **Rationale**: The RP provides an immediate visual of periodic vs. chaotic structure (diagonal lines = periodic, vertical lines = laminar, isolated points = stochastic).  
- **Effort**: Low.

### 1.4 Noise‑Surrogate Testing
Shuffle the posterior time series (e.g., using the Fourier‑based surrogate method) and recompute the MLE.

- **Implementation**:  
  - Generate surrogate time series that preserve the autocorrelation but destroy deterministic structure.  
  - If the MLE of the original series is significantly larger than the surrogate ensemble, chaos is credible.  
- **Rationale**: This is the standard test to rule out that the observed MLE is due to random fluctuations.  
- **Effort**: Medium. Requires careful surrogate generation and statistical testing.

### 1.5 Full Lyapunov Spectrum (Optional, Higher Effort)
If the embedding dimension is sufficiently high, compute all Lyapunov exponents via QR decomposition of the Jacobian of the reconstructed map.

- **Implementation**:  
  - Use a local linear approximation (e.g., neural network or Gaussian process) to estimate the Jacobian of the dynamics.  
  - Apply the standard QR algorithm to extract the spectrum.  
- **Rationale**: The full spectrum gives the rate of divergence in every orthogonal direction, revealing the complete dynamical structure.  
- **Effort**: High. Requires a differentiable model of the dynamics.

---

## 2. Bayesian Engine – Relaxing Independence Assumptions

**Current state**  
- Beta belief states with credible intervals.  
- Dependency‑adjusted likelihood ratios via factor exposure.  
- Evidence weighting by quality.

**What remains** (detailed)

### 2.1 Bayesian Network / Dynamic Factor Model
Replace the heuristic overlap adjustment with a small Bayesian network where latent factors (e.g., `media_attention`, `institutional_signal`) are inferred from the evidence stream.

- **Implementation**:  
  - Define a set of latent factors. Each evidence event provides noisy observations of these factors.  
  - Use a Kalman filter (or particle filter) to update the hidden factor states.  
  - The likelihood ratio for each hypothesis is then a function of the factor states.  
- **Rationale**: This captures genuine dependencies between events that share common factors, rather than using a hand‑crafted overlap score.  
- **Effort**: High. Requires designing the factor model and implementing sequential Monte Carlo or Kalman filtering.

### 2.2 Hierarchical Priors for Source Confidence
Allow the confidence of a given source to be drawn from a Beta distribution, with hyperparameters updated across multiple events from the same source.

- **Implementation**:  
  - Maintain a Beta distribution per source.  
  - When an event from that source arrives, update the hyperparameters based on the observed likelihood ratio.  
- **Rationale**: This learns the reliability of each source over time, rather than using a fixed confidence value.  
- **Effort**: Medium. Straightforward extension of the Beta belief state.

### 2.3 Bayesian Model Averaging (BMA)
Run the model with several competing dependency structures (e.g., independent, single‑factor, two‑factor) and weight the posteriors by their marginal likelihood.

- **Implementation**:  
  - For each model structure, compute the marginal likelihood of the entire evidence sequence.  
  - Combine the posterior distributions using BMA weights proportional to the marginal likelihoods.  
- **Rationale**: This accounts for uncertainty in the dependency structure itself and produces a more robust final posterior.  
- **Effort**: High. Requires computing marginal likelihoods, which may involve numerical integration.

### 2.4 Posterior Predictive Checks
Simulate new evidence events from the current posterior and compare the predicted vs. observed likelihood ratios.

- **Implementation**:  
  - For each hypothesis, sample a likelihood ratio from the posterior distribution.  
  - Use the sampled ratio to generate a synthetic event.  
  - Compare the distribution of synthetic ratios with the actual observed ratios (e.g., via a QQ plot).  
- **Rationale**: This tests whether the model can reproduce the observed data, providing a check for over‑fitting or misspecification.  
- **Effort**: Medium. Straightforward simulation.

---

## 3. Market Integration – Time‑Series and Bayesian Pooling

**Current state**  
- Market signals with liquidity weighting.  
- Market opinion pool (logit‑combination with model probability).

**What remains** (detailed)

### 3.1 Kalman Filter on Market Odds
Model the market’s yes‑odds as a noisy time series with a hidden state (the “true” probability).

- **Implementation**:  
  - Use a Kalman filter with a linear Gaussian state space.  
  - The observation is the reported yes‑odds; the state evolves as a random walk.  
  - The filtered probability is the posterior estimate of the true market belief, with uncertainty.  
- **Rationale**: This separates short‑term noise from genuine shifts in market opinion and provides a smoother, more reliable signal.  
- **Effort**: Medium. Requires careful tuning of process and observation noise.

### 3.2 Bayesian Market Pooling
Replace the fixed logit‑weighted average with a Beta‑mixture model.

- **Implementation**:  
  - Represent the market‑implied belief as a Beta distribution (derived from the Kalman filter output).  
  - The model posterior is also a Beta distribution.  
  - Combine them via a weighted Beta mixture (e.g., using a Dirichlet prior on the mixture weights).  
- **Rationale**: This respects the uncertainty in both sources and produces a pooled posterior that is itself a full distribution, not a point estimate.  
- **Effort**: Medium. Requires a Bayesian mixture framework.

### 3.3 Volatility Adjustment
Add a measure of market uncertainty (e.g., bid‑ask spread, implied volatility) and down‑weight signals during high‑volatility periods.

- **Implementation**:  
  - If data available, use the spread or volume to compute a confidence weight for each market signal.  
  - This weight can be used to scale the reliability in the Kalman filter or opinion pool.  
- **Rationale**: Markets are less informative when they are volatile or illiquid.  
- **Effort**: Low to medium, depending on data availability.

### 3.4 Multi‑Market Arbitrage Detection
Flag inconsistent probabilities across related markets.

- **Implementation**:  
  - For each pair of related markets (e.g., “AI safety bill” and “Vatican guidance”), compute the implied joint probability and check for violation of the probability axioms.  
  - Flag significant deviations as potential noise or manipulation.  
- **Rationale**: Inconsistent markets indicate that at least one signal is unreliable.  
- **Effort**: Low.

---

## 4. Narrative PCA – Real‑Time and Automated

**Current state**  
- Dynamic PCA from `narrative_features` on events.  
- Default matrix fallback.

**What remains** (detailed)

### 4.1 Incremental PCA
Update the PCA as new events arrive.

- **Implementation**:  
  - Use `sklearn.decomposition.IncrementalPCA`.  
  - Each new event’s feature vector is added to the rotation, and the components and loadings are updated.  
- **Rationale**: This allows the narrative geometry to evolve in real‑time without recomputing from scratch.  
- **Effort**: Low.

### 4.2 NLP Feature Extraction
Replace hand‑crafted `narrative_features` with embeddings from a sentence‑transformer model.

- **Implementation**:  
  - Use a pre‑trained model (e.g., `all‑MiniLM‑L6‑v2` from `sentence‑transformers`).  
  - Optionally, project the high‑dimensional embeddings onto the existing 7 dimensions using a linear regression or a small neural network.  
- **Rationale**: This removes the biggest bottleneck – manual feature engineering – and makes the PCA component scalable to new domains.  
- **Effort**: Medium. Requires integrating a NLP model and handling missing embeddings.

### 4.3 Topic Modeling
Use BERTopic or LDA to discover latent narrative dimensions from event text.

- **Implementation**:  
  - On the event titles/rationales, run BERTopic to extract topics.  
  - Each event gets a topic‑distribution vector that can be used as the narrative feature vector.  
- **Rationale**: This provides an unsupervised, data‑driven alternative to hand‑crafted dimensions.  
- **Effort**: Medium. Requires careful tuning of topic models and alignment with existing hypotheses.

### 4.4 Temporal PCA
Run PCA on sliding windows to see how the narrative axes rotate over time.

- **Implementation**:  
  - For each window, compute the PCA and record the explained variance and loadings.  
  - Plot the angle of each loading vector over time to visualise axis rotation.  
- **Rationale**: This reveals whether the narrative structure is stable or evolving, which is itself a diagnostic.  
- **Effort**: Low.

---

## 5. Validation & Calibration Framework

**Current state**  
- Sensitivity analysis over priors, likelihood scales, confidence scales.

**What remains** (detailed)

### 5.1 Backtesting on Synthetic Data
Generate a ground‑truth sequence where the “true” hypothesis is known, then run the tracker and measure calibration.

- **Implementation**:  
  - Create a hidden Markov model where the hidden state is the true hypothesis.  
  - Generate evidence events with known likelihood ratios conditional on the hidden state.  
  - Run the tracker and compare posteriors to the true state.  
  - Compute metrics: Brier score, log‑loss, calibration error.  
- **Rationale**: This provides quantitative evidence that the tracker works as intended.  
- **Effort**: Medium. Requires designing the synthetic generator and evaluation metrics.

### 5.2 Posterior Calibration Curve
For each posterior bin, plot the observed frequency of the hypothesis being “true” in synthetic tests.

- **Implementation**:  
  - Bin posteriors (0–10%, 10–20%, …).  
  - In each bin, compute the fraction of times the hypothesis was true.  
  - Plot against the bin’s midpoint. A well‑calibrated model should lie on the diagonal.  
- **Rationale**: This is the gold‑standard for Bayesian models; it checks that the posteriors are honest.  
- **Effort**: Low.

### 5.3 Stability Classification Validation
On synthetic trajectories with known Lyapunov exponents, check how often the classification (stable/unstable/saddle) matches.

- **Implementation**:  
  - Generate trajectories from known dynamical systems (e.g., logistic map, Hénon map, random walk).  
  - Run the stability diagnostic and compare the output to the true Lyapunov exponent.  
  - Compute confusion matrices for the classification.  
- **Rationale**: This validates the stability diagnostic against known ground truth.  
- **Effort**: Medium. Requires careful simulation of dynamical systems.

### 5.4 Robustness to Misspecification
Run the tracker with misspecified priors or dependency structures and measure performance degradation.

- **Implementation**:  
  - Systematically vary the prior, likelihood ratios, and confidence values away from the true values.  
  - Measure how much the posterior diverges from the optimal posterior.  
- **Rationale**: This quantifies how robust the tool is to common user errors.  
- **Effort**: Medium.

