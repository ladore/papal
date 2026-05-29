from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Literal, cast

try:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    np = None
    PCA = None
    StandardScaler = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


EvidenceQuality = Literal["primary", "secondary", "market", "hypothetical", "unverified"]
Direction = Literal["supports", "opposes", "neutral"]
MORAL_LEGITIMACY_AXIS = "moral_legitimacy_axis"
POLICY_MARKET_AXIS = "policy_market_axis"
MORAL_LEGITIMACY_LABEL = "Moral Legitimacy Axis"
POLICY_MARKET_LABEL = "Policy / Market Action Axis"


@dataclass(frozen=True)
class Hypothesis:
    key: str
    label: str
    prior: float
    description: str


@dataclass(frozen=True)
class EvidenceEvent:
    date: datetime
    title: str
    quality: EvidenceQuality
    source: str
    rationale: str
    likelihood_ratios: dict[str, float]
    confidence: float = 1.0
    dependency_factors: dict[str, float] = field(default_factory=dict)
    narrative_features: dict[str, float] | None = None

    @classmethod
    def from_date(
        cls,
        date_str: str,
        title: str,
        quality: EvidenceQuality,
        source: str,
        rationale: str,
        likelihood_ratios: dict[str, float],
        confidence: float = 1.0,
        dependency_factors: dict[str, float] | None = None,
        narrative_features: dict[str, float] | None = None,
    ) -> "EvidenceEvent":
        return cls(
            date=datetime.strptime(date_str, "%Y-%m-%d"),
            title=title,
            quality=quality,
            source=source,
            rationale=rationale,
            likelihood_ratios=likelihood_ratios,
            confidence=confidence,
            dependency_factors=dependency_factors or {},
            narrative_features=narrative_features,
        )


@dataclass(frozen=True)
class MarketSignal:
    date: datetime
    question: str
    yes_odds: float
    volume_usd: float | None
    relevance: dict[str, float]
    source: str = "Polymarket"

    @classmethod
    def from_date(
        cls,
        date_str: str,
        question: str,
        yes_odds: float,
        volume_usd: float | None,
        relevance: dict[str, float],
        source: str = "Polymarket",
    ) -> "MarketSignal":
        return cls(
            date=datetime.strptime(date_str, "%Y-%m-%d"),
            question=question,
            yes_odds=yes_odds,
            volume_usd=volume_usd,
            relevance=relevance,
            source=source,
        )

    @property
    def liquidity_weight(self) -> float:
        if self.volume_usd is None:
            return 0.25
        if self.volume_usd < 25_000:
            return 0.35
        if self.volume_usd < 100_000:
            return 0.55
        if self.volume_usd < 500_000:
            return 0.75
        return 1.0


@dataclass
class BeliefState:
    hypothesis: str
    alpha: float
    beta: float

    @classmethod
    def from_prior(
        cls, hypothesis: str, prior: float, prior_strength: float = 12.0
    ) -> "BeliefState":
        return cls(
            hypothesis=hypothesis,
            alpha=max(prior * prior_strength, 1e-6),
            beta=max((1 - prior) * prior_strength, 1e-6),
        )

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def interval(self, z_score: float = 1.64) -> tuple[float, float]:
        radius = z_score * math.sqrt(self.variance)
        return clamp_probability(self.mean - radius), clamp_probability(self.mean + radius)

    def observe(self, likelihood_ratio: float, weight: float) -> None:
        support_probability = likelihood_ratio / (1 + likelihood_ratio)
        bounded_weight = max(weight, 0.0)
        self.alpha += support_probability * bounded_weight
        self.beta += (1 - support_probability) * bounded_weight


@dataclass
class PosteriorPoint:
    date: datetime
    hypothesis: str
    posterior: float
    driver: str
    likelihood_ratio: float


@dataclass
class NarrativePCAResult:
    labels: list[str]
    dimensions: list[str]
    components: list[list[float]]
    explained_variance: list[float]
    loadings: list[dict[str, float | str]]


@dataclass
class ChaosStabilityResult:
    classification: str
    confidence: str
    lyapunov_proxy: float | None
    step_rows: list[dict[str, object]]
    hypothesis_rows: list[dict[str, object]]
    explanation: str
    recurrence_rows: list[dict[str, object]] = field(default_factory=list)
    lyapunov_rows: list[dict[str, object]] = field(default_factory=list)
    dimension_rows: list[dict[str, object]] = field(default_factory=list)
    surrogate_rows: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class DynamicalAnalysisConfig:
    embedding_dimension: int = 2
    delay: int = 1
    recurrence_threshold: float = 0.08
    min_points_for_rqa: int = 5
    min_points_for_mle: int = 8
    min_points_for_correlation_dimension: int = 8
    theiler_window: int = 1
    max_lyapunov_horizon: int = 4
    surrogate_runs: int = 24
    surrogate_seed: int = 17


@dataclass(frozen=True)
class SensitivityConfig:
    prior_shifts: tuple[float, ...] = (-0.05, 0.0, 0.05)
    likelihood_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    confidence_scales: tuple[float, ...] = (0.85, 1.0, 1.15)


@dataclass
class SensitivityResult:
    rows: list[dict[str, object]]
    classification_rows: list[dict[str, object]]
    summary: str


@dataclass(frozen=True)
class MarketFilterConfig:
    process_noise: float = 0.0025
    observation_noise_floor: float = 0.015
    initial_variance: float = 0.08


@dataclass
class SyntheticValidationResult:
    rows: list[dict[str, object]]
    summary: str


class ResonanceBayesianTracker:
    """
    A single-file evidence ledger for updating multiple narrative hypotheses.

    The model is intentionally modest: it does not prove causality. It keeps a
    disciplined, inspectable trail of priors, evidence, likelihood ratios,
    market signals, and narrative geometry.
    """

    def __init__(self, hypotheses: list[Hypothesis] | None = None, verbose: bool = True):
        self.hypotheses = hypotheses or self.default_hypotheses()
        self.verbose = verbose
        self.posteriors = {h.key: h.prior for h in self.hypotheses}
        self.belief_states = {
            h.key: BeliefState.from_prior(h.key, h.prior) for h in self.hypotheses
        }
        self.factor_exposure: dict[str, float] = {}
        self.events: list[EvidenceEvent] = []
        self.market_signals: list[MarketSignal] = []
        self.posterior_history: list[PosteriorPoint] = []

    @staticmethod
    def default_hypotheses() -> list[Hypothesis]:
        return [
            Hypothesis(
                key="governance_influence",
                label="Institutional AI governance influence",
                prior=0.45,
                description="Vatican or adjacent religious institutions materially affect AI governance discourse.",
            ),
            Hypothesis(
                key="symbolic_legitimation",
                label="Symbolic legitimation",
                prior=0.62,
                description="The event primarily lends moral language and public legitimacy to AI actors.",
            ),
            Hypothesis(
                key="anthropic_brand_benefit",
                label="Anthropic brand benefit",
                prior=0.58,
                description="Anthropic receives reputational upside independent of hard policy movement.",
            ),
            Hypothesis(
                key="low_policy_impact",
                label="Low direct policy impact",
                prior=0.55,
                description="The event has little direct effect on law, regulation, or market structure.",
            ),
        ]

    def add_event(self, event: EvidenceEvent) -> None:
        self.events.append(event)
        for hypothesis in self.hypotheses:
            raw_lr = event.likelihood_ratios.get(hypothesis.key, 1.0)
            dependency_adjusted_lr = self._dependency_adjusted_likelihood_ratio(
                raw_lr, event.dependency_factors
            )
            weighted_lr = self._temper_likelihood_ratio(
                dependency_adjusted_lr, event.confidence
            )
            self.posteriors[hypothesis.key] = self._bayes_update(
                self.posteriors[hypothesis.key], weighted_lr
            )
            self.belief_states[hypothesis.key].observe(
                weighted_lr, evidence_weight(event)
            )
            self.posterior_history.append(
                PosteriorPoint(
                    date=event.date,
                    hypothesis=hypothesis.key,
                    posterior=self.posteriors[hypothesis.key],
                    driver=event.title,
                    likelihood_ratio=weighted_lr,
                )
            )

        for factor, exposure in event.dependency_factors.items():
            self.factor_exposure[factor] = self.factor_exposure.get(factor, 0.0) + exposure

        if self.verbose:
            print(f"[{event.date.date()}] {event.title}")
            print_console_note(
                "Posterior update",
                "`posterior` is the fast belief update; `credible_interval` shows uncertainty.",
            )
            posterior_rows = self.posterior_table()
            if posterior_rows:
                top = posterior_rows[0]
                print_console_note(
                    "Current leader",
                    f"{top['hypothesis']} at {top['posterior']} "
                    f"({top['interpretation']}).",
                )
            print(format_table(posterior_rows))

    def add_simple_event(
        self,
        date_str: str,
        description: str,
        likelihood_ratio: float,
        hypothesis: str = "symbolic_legitimation",
    ) -> None:
        """Compatibility helper for the original single-hypothesis style."""
        self.add_event(
            EvidenceEvent.from_date(
                date_str=date_str,
                title=description,
                quality="hypothetical",
                source="manual entry",
                rationale="Legacy event format; likelihood ratio applies to one selected hypothesis.",
                likelihood_ratios={hypothesis: likelihood_ratio},
                confidence=0.75,
            )
        )

    def add_market_signal(self, signal: MarketSignal) -> None:
        self.market_signals.append(signal)
        print(
            f"Market: '{signal.question}' | Yes: {signal.yes_odds:.1%} | "
            f"liquidity weight: {signal.liquidity_weight:.2f} "
            f"({liquidity_phrase(signal.liquidity_weight)})"
        )

    def add_polymarket_bet(
        self,
        date_str: str,
        question: str,
        yes_odds: float,
        volume: str | float | None,
        relevance: dict[str, float] | None = None,
    ) -> None:
        self.add_market_signal(
            MarketSignal.from_date(
                date_str=date_str,
                question=question,
                yes_odds=yes_odds,
                volume_usd=self._parse_volume(volume),
                relevance=relevance or {},
            )
        )

    def posterior_table(self) -> list[dict[str, object]]:
        rows = []
        for hypothesis in self.hypotheses:
            belief = self.belief_states[hypothesis.key]
            lower, upper = belief.interval()
            rows.append(
                {
                    "hypothesis": hypothesis.key,
                    "posterior": round(self.posteriors[hypothesis.key], 4),
                    "interpretation": probability_phrase(self.posteriors[hypothesis.key]),
                    "belief_mean": round(belief.mean, 4),
                    "credible_interval": f"{lower:.3f}-{upper:.3f}",
                    "uncertainty": uncertainty_phrase(upper - lower),
                    "label": hypothesis.label,
                }
            )
        return sorted(rows, key=lambda row: cast(float, row["posterior"]), reverse=True)

    def event_table(self) -> list[dict[str, object]]:
        return [
            {
                "date": event.date.date().isoformat(),
                "title": event.title,
                "quality": event.quality,
                "confidence": event.confidence,
                "dependencies": format_factor_map(event.dependency_factors),
                "narrative_features": format_factor_map(event.narrative_features or {}),
                "source": event.source,
                "rationale": event.rationale,
            }
            for event in self.events
        ]

    def market_table(self) -> list[dict[str, object]]:
        rows = []
        for signal in self.market_signals:
            weighted_signal = (signal.yes_odds - 0.5) * signal.liquidity_weight
            rows.append(
                {
                    "date": signal.date.date().isoformat(),
                    "question": signal.question,
                    "yes_odds": round(signal.yes_odds, 4),
                    "market_read": probability_phrase(signal.yes_odds),
                    "volume_usd": signal.volume_usd,
                    "liquidity_weight": round(signal.liquidity_weight, 3),
                    "weighted_signal": round(weighted_signal, 4),
                    "signal_read": market_signal_phrase(weighted_signal),
                    "source": signal.source,
                }
            )
        return rows

    def history_table(self) -> list[dict[str, object]]:
        return [
            {
                "date": point.date.date().isoformat(),
                "hypothesis": point.hypothesis,
                "posterior": round(point.posterior, 4),
                "likelihood_ratio": round(point.likelihood_ratio, 4),
                "driver": point.driver,
            }
            for point in self.posterior_history
        ]

    def plot_probability_timeline(self, output_path: str | None = None) -> None:
        if plt is None:
            print("Skipping timeline plot because matplotlib is not installed.")
            return
        if not self.posterior_history:
            print("No posterior history to plot.")
            return

        history = self.history_table()
        plt.figure(figsize=(11, 7))
        hypotheses = sorted({cast(str, row["hypothesis"]) for row in history})
        date_labels = sorted({cast(str, row["date"]) for row in history})
        x_positions = {date_label: index for index, date_label in enumerate(date_labels)}
        for hypothesis in hypotheses:
            subset = [row for row in history if row["hypothesis"] == hypothesis]
            plt.plot(
                [x_positions[cast(str, row["date"])] for row in subset],
                [cast(float, row["posterior"]) for row in subset],
                marker="o",
                linewidth=2,
                label=hypothesis,
            )

        plt.title("Bayesian Probability Timeline: Vatican / AI Narrative Hypotheses")
        plt.ylabel("Posterior probability")
        plt.xlabel("Date")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(list(range(len(date_labels))), date_labels, rotation=35)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=160)
            print(f"Saved timeline plot to {output_path}")
        else:
            plt.show()

    def show_polymarket_summary(self) -> None:
        if not self.market_signals:
            print("\nNo market signals registered.")
            return
        print("\n=== Market Signal Summary ===")
        print_console_note(
            "What this is",
            "Raw market odds are treated as outside signals. Thin or missing-volume markets get "
            "lower weight, so they can inform the model without overpowering it.",
        )
        print(format_table(self.market_table()))
        opinion_rows = self.market_opinion_pool_table()
        if opinion_rows:
            print("\n=== Market Opinion Pool ===")
            print_console_note(
                "How to read it",
                "`pooled_probability` blends the model posterior with relevant market odds in "
                "log-odds space. `shift` shows how much markets pull the model up or down.",
            )
            print(format_table(opinion_rows))
        filtered_rows = self.market_kalman_table()
        if filtered_rows:
            print("\n=== Market Kalman Filter ===")
            print_console_note(
                "How to read it",
                "The filter estimates a smoother hidden market belief from noisy odds. With one "
                "observation it mostly mirrors the market; with a time series it becomes more useful.",
            )
            print(format_table(filtered_rows))
        arbitrage_rows = self.market_arbitrage_table()
        if arbitrage_rows:
            print("\n=== Market Consistency Checks ===")
            print_console_note(
                "Why this matters",
                "Large gaps between related pooled beliefs are not automatic errors, but they mark "
                "places where relevance weights or market assumptions deserve inspection.",
            )
            print(format_table(arbitrage_rows))

    def market_opinion_pool_table(self, model_weight: float = 1.0) -> list[dict[str, object]]:
        rows = []
        for hypothesis in self.hypotheses:
            weighted_market_logits = []
            total_market_weight = 0.0
            for signal in self.market_signals:
                relevance = signal.relevance.get(hypothesis.key, 0.0)
                reliability = signal.liquidity_weight * relevance
                if reliability <= 0:
                    continue
                weighted_market_logits.append(reliability * logit(signal.yes_odds))
                total_market_weight += reliability

            if total_market_weight <= 0:
                continue

            model_probability = self.posteriors[hypothesis.key]
            combined_logit = (
                model_weight * logit(model_probability) + sum(weighted_market_logits)
            ) / (model_weight + total_market_weight)
            pooled_probability = inverse_logit(combined_logit)
            rows.append(
                {
                    "hypothesis": hypothesis.key,
                    "model_probability": round(model_probability, 4),
                    "pooled_probability": round(pooled_probability, 4),
                    "market_weight": round(total_market_weight, 4),
                    "pooled_read": probability_phrase(pooled_probability),
                    "shift": round(pooled_probability - model_probability, 4),
                }
            )
        return rows

    def market_kalman_table(
        self, config: MarketFilterConfig | None = None
    ) -> list[dict[str, object]]:
        config = config or MarketFilterConfig()
        rows = []
        for question, signals in group_market_signals(self.market_signals).items():
            ordered = sorted(signals, key=lambda signal: signal.date)
            estimates = kalman_filter_market_odds(ordered, config)
            if not estimates:
                continue
            latest = estimates[-1]
            rows.append(
                {
                    "question": question,
                    "observations": len(ordered),
                    "latest_observed": round(ordered[-1].yes_odds, 4),
                    "filtered_probability": round(latest["state"], 4),
                    "variance": round(latest["variance"], 5),
                    "read": probability_phrase(cast(float, latest["state"])),
                }
            )
        return rows

    def market_arbitrage_table(self) -> list[dict[str, object]]:
        pooled = self.market_opinion_pool_table()
        rows = []
        for left_index in range(len(pooled)):
            for right_index in range(left_index + 1, len(pooled)):
                left = pooled[left_index]
                right = pooled[right_index]
                difference = abs(
                    cast(float, left["pooled_probability"])
                    - cast(float, right["pooled_probability"])
                )
                if difference < 0.30:
                    continue
                rows.append(
                    {
                        "left": left["hypothesis"],
                        "right": right["hypothesis"],
                        "probability_gap": round(difference, 4),
                        "read": "large cross-market/model gap; inspect relevance mapping",
                    }
                )
        return rows

    def export_recurrence_matrix(
        self,
        output_path: str,
        config: DynamicalAnalysisConfig | None = None,
    ) -> None:
        config = config or DynamicalAnalysisConfig()
        embedded = self._embedded_state_vectors(self._posterior_snapshots(), config)
        matrix = recurrence_matrix(embedded, config.recurrence_threshold)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(",".join(str(value) for value in row) for row in matrix) + "\n",
            encoding="utf-8",
        )
        print(f"Saved recurrence matrix to {path}")

    def plot_recurrence_matrix(
        self,
        output_path: str | None = None,
        config: DynamicalAnalysisConfig | None = None,
    ) -> None:
        if plt is None:
            print("Skipping recurrence plot because matplotlib is not installed.")
            return
        config = config or DynamicalAnalysisConfig()
        embedded = self._embedded_state_vectors(self._posterior_snapshots(), config)
        matrix = recurrence_matrix(embedded, config.recurrence_threshold)
        if not matrix:
            print("No recurrence matrix to plot.")
            return
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="Greys", interpolation="nearest")
        plt.title("Recurrence Plot")
        plt.xlabel("Embedded state index")
        plt.ylabel("Embedded state index")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=160)
            print(f"Saved recurrence plot to {output_path}")
        else:
            plt.show()

    def run_sensitivity_analysis(
        self, config: SensitivityConfig | None = None
    ) -> SensitivityResult:
        config = config or SensitivityConfig()
        if not self.events:
            result = SensitivityResult([], [], "No evidence events available for sensitivity analysis.")
            self.print_sensitivity_result(result)
            return result

        outcomes: list[dict[str, object]] = []
        classifications: dict[str, int] = {}
        for prior_shift in config.prior_shifts:
            shifted_hypotheses = [
                replace(
                    hypothesis,
                    prior=clamp_probability(hypothesis.prior + prior_shift),
                )
                for hypothesis in self.hypotheses
            ]
            for likelihood_scale in config.likelihood_scales:
                for confidence_scale in config.confidence_scales:
                    trial = ResonanceBayesianTracker(shifted_hypotheses, verbose=False)
                    for event in self.events:
                        scaled_event = replace(
                            event,
                            likelihood_ratios={
                                key: scale_likelihood_ratio(value, likelihood_scale)
                                for key, value in event.likelihood_ratios.items()
                            },
                            confidence=clamp_unit_interval(
                                event.confidence * confidence_scale
                            ),
                        )
                        trial.add_event(scaled_event)
                    chaos = trial.run_chaos_stability_analysis()
                    classifications[chaos.classification] = (
                        classifications.get(chaos.classification, 0) + 1
                    )
                    for hypothesis in trial.hypotheses:
                        outcomes.append(
                            {
                                "hypothesis": hypothesis.key,
                                "posterior": trial.posteriors[hypothesis.key],
                                "classification": chaos.classification,
                            }
                        )

        rows = []
        for hypothesis in self.hypotheses:
            values = [
                cast(float, outcome["posterior"])
                for outcome in outcomes
                if outcome["hypothesis"] == hypothesis.key
            ]
            if not values:
                continue
            rows.append(
                {
                    "hypothesis": hypothesis.key,
                    "min_posterior": round(min(values), 4),
                    "max_posterior": round(max(values), 4),
                    "spread": round(max(values) - min(values), 4),
                    "robustness": sensitivity_spread_phrase(max(values) - min(values)),
                }
            )

        total_runs = sum(classifications.values())
        classification_rows = [
            {
                "classification": classification,
                "count": count,
                "share": round(count / total_runs, 4) if total_runs else 0.0,
            }
            for classification, count in sorted(
                classifications.items(), key=lambda item: item[1], reverse=True
            )
        ]
        summary = sensitivity_summary(rows, classification_rows)
        result = SensitivityResult(rows, classification_rows, summary)
        self.print_sensitivity_result(result)
        return result

    def print_sensitivity_result(self, result: SensitivityResult) -> None:
        print("\n=== Sensitivity Analysis ===")
        print_console_note(
            "What this tests",
            "The model reruns under shifted priors, likelihood ratios, and confidence values. "
            "Small posterior spreads mean the conclusion is robust; large spreads mean it is fragile.",
        )
        print(result.summary)
        if result.rows:
            print("\nPosterior robustness:")
            print_console_note(
                "Column guide",
                "`min_posterior` and `max_posterior` are the range seen across perturbations. "
                "`spread` is the size of that range.",
            )
            print(format_table(result.rows))
        if result.classification_rows:
            print("\nClassification robustness:")
            print_console_note(
                "Column guide",
                "`share` is how often each stability label appeared across sensitivity runs.",
            )
            print(format_table(result.classification_rows))

    def run_chaos_stability_analysis(
        self, config: DynamicalAnalysisConfig | None = None
    ) -> ChaosStabilityResult:
        config = config or DynamicalAnalysisConfig()
        snapshots = self._posterior_snapshots()
        if len(snapshots) < 3:
            result = ChaosStabilityResult(
                classification="insufficient trajectory",
                confidence="low",
                lyapunov_proxy=None,
                step_rows=[],
                hypothesis_rows=[],
                recurrence_rows=[],
                explanation=(
                    "Need at least two event-to-event movements after the initial prior "
                    "to say anything useful about stability."
                ),
            )
            if self.verbose:
                self.print_chaos_stability_result(result)
            return result

        hypothesis_keys = [hypothesis.key for hypothesis in self.hypotheses]
        deltas = [
            [
                snapshots[index][1][key] - snapshots[index - 1][1][key]
                for key in hypothesis_keys
            ]
            for index in range(1, len(snapshots))
        ]
        step_norms = [euclidean_norm(delta) for delta in deltas]
        ratios = [
            step_norms[index] / step_norms[index - 1]
            for index in range(1, len(step_norms))
            if step_norms[index - 1] > 0
        ]
        lyapunov_proxy = (
            sum(math.log(max(ratio, 1e-9)) for ratio in ratios) / len(ratios)
            if ratios
            else None
        )

        step_rows = []
        for index, norm in enumerate(step_norms, start=1):
            ratio = None if index == 1 or step_norms[index - 2] == 0 else norm / step_norms[index - 2]
            step_rows.append(
                {
                    "from": snapshots[index - 1][0],
                    "to": snapshots[index][0],
                    "movement_norm": round(norm, 4),
                    "growth_ratio": round(ratio, 4) if ratio is not None else "N/A",
                    "read": stability_step_phrase(ratio),
                }
            )

        hypothesis_rows = []
        local_rates = []
        for axis_index, key in enumerate(hypothesis_keys):
            series = [delta[axis_index] for delta in deltas]
            rate = local_axis_growth_rate(series)
            if rate is not None:
                local_rates.append(rate)
            hypothesis_rows.append(
                {
                    "hypothesis": key,
                    "latest_delta": round(series[-1], 4),
                    "local_growth": round(rate, 4) if rate is not None else "N/A",
                    "stability_read": local_stability_phrase(rate),
                    "direction_read": delta_direction_phrase(series),
                }
            )

        classification = classify_stability(step_norms, local_rates, lyapunov_proxy)
        confidence = chaos_confidence_phrase(len(step_norms), ratios)
        recurrence_rows = self._recurrence_rows(snapshots, config)
        embedded = self._embedded_state_vectors(snapshots, config)
        lyapunov_rows = lyapunov_diagnostic_rows(embedded, config)
        dimension_rows = correlation_dimension_rows(embedded, config)
        surrogate_rows = surrogate_lyapunov_rows(
            embedded,
            config,
            cast(float | None, extract_metric_value(lyapunov_rows, "rosenstein_mle")),
        )
        explanation = chaos_classification_explanation(classification, lyapunov_proxy)
        result = ChaosStabilityResult(
            classification=classification,
            confidence=confidence,
            lyapunov_proxy=lyapunov_proxy,
            step_rows=step_rows,
            hypothesis_rows=hypothesis_rows,
            recurrence_rows=recurrence_rows,
            lyapunov_rows=lyapunov_rows,
            dimension_rows=dimension_rows,
            surrogate_rows=surrogate_rows,
            explanation=explanation,
        )
        if self.verbose:
            self.print_chaos_stability_result(result)
        return result

    def print_chaos_stability_result(self, result: ChaosStabilityResult) -> None:
        print("\n=== Chaos / Stability Diagnostic ===")
        print_console_note(
            "Plain meaning",
            "This treats the vector of hypothesis probabilities as a moving state. Shrinking "
            "movement suggests convergence; growing movement suggests instability; mixed axes can "
            "look saddle-like.",
        )
        print(f"Classification: {result.classification}")
        print(f"Confidence: {result.confidence}")
        if result.lyapunov_proxy is not None:
            print(f"Lyapunov-style proxy: {result.lyapunov_proxy:.4f}")
        print(result.explanation)
        if result.step_rows:
            print("\nEvent-to-event movement:")
            print_console_note(
                "How to read it",
                "`movement_norm` is the size of the jump in posterior space. `growth_ratio` below "
                "1 means the next jump got smaller; above 1 means it got larger.",
            )
            print(format_table(result.step_rows))
        if result.hypothesis_rows:
            print("\nHypothesis-level local behavior:")
            print_console_note(
                "How to read it",
                "This breaks the movement down by hypothesis, so you can see which beliefs are "
                "still moving even if the overall system looks stable.",
            )
            print(format_table(result.hypothesis_rows))
        if result.recurrence_rows:
            print("\nRecurrence quantification:")
            print_console_note(
                "How to read it",
                "Recurrence asks whether the system revisits similar states. The current demo is "
                "usually too short for this to be rigorous, and the table says so explicitly.",
            )
            print(format_table(result.recurrence_rows))
        if result.lyapunov_rows:
            print("\nRosenstein Lyapunov diagnostic:")
            print_console_note(
                "How to read it",
                "A positive MLE is chaos-compatible, a negative one is convergence-compatible, "
                "and near zero suggests marginal or periodic dynamics. Enough data is required.",
            )
            print(format_table(result.lyapunov_rows))
        if result.dimension_rows:
            print("\nCorrelation dimension diagnostic:")
            print_console_note(
                "How to read it",
                "D2 estimates attractor complexity. Low values suggest low-dimensional structure; "
                "high or unstable values suggest noise or insufficient data.",
            )
            print(format_table(result.dimension_rows))
        if result.surrogate_rows:
            print("\nSurrogate test:")
            print_console_note(
                "How to read it",
                "Surrogates ask whether apparent divergence is stronger than a shuffled baseline. "
                "This helps avoid mistaking noise for chaos.",
            )
            print(format_table(result.surrogate_rows))

    def _embedded_state_vectors(
        self,
        snapshots: list[tuple[str, dict[str, float]]],
        config: DynamicalAnalysisConfig,
    ) -> list[list[float]]:
        state_vectors = [
            [state[hypothesis.key] for hypothesis in self.hypotheses]
            for _, state in snapshots
        ]
        return takens_embedding_vectors(
            state_vectors, config.embedding_dimension, config.delay
        )

    def _recurrence_rows(
        self,
        snapshots: list[tuple[str, dict[str, float]]],
        config: DynamicalAnalysisConfig,
    ) -> list[dict[str, object]]:
        embedded = self._embedded_state_vectors(snapshots, config)
        if len(embedded) < config.min_points_for_rqa:
            return [
                {
                    "metric": "rqa_status",
                    "value": "insufficient points",
                    "read": (
                        f"need at least {config.min_points_for_rqa} embedded points; "
                        f"have {len(embedded)}"
                    ),
                }
            ]
        metrics = recurrence_quantification(embedded, config.recurrence_threshold)
        return [
            {
                "metric": key,
                "value": round(value, 4),
                "read": rqa_metric_phrase(key, value),
            }
            for key, value in metrics.items()
        ]

    def _posterior_snapshots(self) -> list[tuple[str, dict[str, float]]]:
        snapshots = [
            (
                "initial_prior",
                {hypothesis.key: hypothesis.prior for hypothesis in self.hypotheses},
            )
        ]
        hypothesis_count = len(self.hypotheses)
        if hypothesis_count == 0:
            return snapshots

        state = snapshots[0][1].copy()
        for start in range(0, len(self.posterior_history), hypothesis_count):
            chunk = self.posterior_history[start : start + hypothesis_count]
            if len(chunk) < hypothesis_count:
                break
            for point in chunk:
                state[point.hypothesis] = point.posterior
            label = chunk[-1].date.date().isoformat()
            snapshots.append((label, state.copy()))
        return snapshots

    def _event_narrative_matrix(self) -> tuple[list[str], list[str], list[list[float]]] | None:
        feature_events = [event for event in self.events if event.narrative_features]
        if len(feature_events) < 3:
            return None

        dimensions = sorted(
            {
                feature
                for event in feature_events
                for feature in cast(dict[str, float], event.narrative_features).keys()
            }
        )
        labels = [event.title[:32].replace(" ", "_") for event in feature_events]
        data = [
            [
                cast(dict[str, float], event.narrative_features).get(dimension, 0.0)
                for dimension in dimensions
            ]
            for event in feature_events
        ]
        return labels, dimensions, data

    def run_narrative_pca(self) -> NarrativePCAResult | None:
        if np is None or PCA is None or StandardScaler is None:
            print("Skipping narrative PCA because numpy or scikit-learn is not installed.")
            return None

        event_matrix = self._event_narrative_matrix()
        if event_matrix is None:
            dimensions, labels, matrix = default_narrative_matrix()
        else:
            labels, dimensions, matrix = event_matrix

        data = np.array(matrix)

        scaled = StandardScaler().fit_transform(data)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        component_loadings = pca.components_.T.tolist()
        loading_rows = [
            {
                "dimension": dimension,
                MORAL_LEGITIMACY_AXIS: cast(float, component_loadings[index][0]),
                POLICY_MARKET_AXIS: cast(float, component_loadings[index][1]),
            }
            for index, dimension in enumerate(dimensions)
        ]

        result = NarrativePCAResult(
            labels=labels,
            dimensions=dimensions,
            components=components.tolist(),
            explained_variance=pca.explained_variance_ratio_.tolist(),
            loadings=loading_rows,
        )
        self.print_pca_result(result)
        return result

    def run_temporal_pca(self, window_size: int = 3) -> list[dict[str, object]]:
        if np is None or PCA is None or StandardScaler is None:
            print("Skipping temporal PCA because numpy or scikit-learn is not installed.")
            return []
        event_matrix = self._event_narrative_matrix()
        if event_matrix is None:
            print("Skipping temporal PCA because event narrative features are insufficient.")
            return []
        labels, dimensions, matrix = event_matrix
        if window_size < 2:
            raise ValueError("Window size must be at least 2.")
        if len(matrix) < window_size:
            print("Skipping temporal PCA because the window is larger than the event matrix.")
            return []

        rows = []
        previous_loading: list[float] | None = None
        for start in range(0, len(matrix) - window_size + 1):
            window = matrix[start : start + window_size]
            scaled = StandardScaler().fit_transform(np.array(window))
            pca = PCA(n_components=2)
            pca.fit_transform(scaled)
            loading = cast(list[float], pca.components_[0].tolist())
            rotation = (
                vector_angle(previous_loading, loading)
                if previous_loading is not None
                else None
            )
            rows.append(
                {
                    "window": f"{labels[start]} -> {labels[start + window_size - 1]}",
                    "variance_axis_1": round(float(pca.explained_variance_ratio_[0]), 4),
                    "variance_axis_2": round(float(pca.explained_variance_ratio_[1]), 4),
                    "axis_rotation_deg": round(rotation, 2) if rotation is not None else "N/A",
                    "read": temporal_pca_phrase(rotation),
                    "dimensions": len(dimensions),
                }
            )
            previous_loading = loading
        print("\n=== Temporal PCA ===")
        print_console_note(
            "What this shows",
            "Temporal PCA compares short windows of narrative features. Large axis rotation means "
            "the narrative geometry is changing, not merely moving along a fixed axis.",
        )
        print(format_table(rows))
        return rows

    def print_pca_result(self, result: NarrativePCAResult) -> None:
        print("\n=== Narrative PCA ===")
        print_console_note(
            "Plain meaning",
            "PCA compresses narrative features into two readable axes. Axis scores show where "
            "each event sits relative to the center of the narrative map.",
        )
        print("Explained variance:", [round(value, 3) for value in result.explained_variance])
        pca_rows = [
            {
                "label": label,
                MORAL_LEGITIMACY_AXIS: round(component[0], 3),
                "moral_legitimacy_position": moral_legitimacy_score_phrase(component[0]),
                POLICY_MARKET_AXIS: round(component[1], 3),
                "policy_market_position": policy_market_score_phrase(component[1]),
            }
            for label, component in zip(result.labels, result.components)
        ]
        print(format_table(pca_rows))

        print("\n=== PCA Loadings ===")
        print_console_note(
            "How to read it",
            "Loadings explain what each axis is made of. Strong positive or negative pulls mean "
            "that feature strongly defines that side of the axis.",
        )
        rounded_loadings = [
            {
                "dimension": row["dimension"],
                MORAL_LEGITIMACY_AXIS: round(cast(float, row[MORAL_LEGITIMACY_AXIS]), 3),
                "moral_legitimacy_read": pca_loading_phrase(
                    cast(float, row[MORAL_LEGITIMACY_AXIS])
                ),
                POLICY_MARKET_AXIS: round(cast(float, row[POLICY_MARKET_AXIS]), 3),
                "policy_market_read": pca_loading_phrase(
                    cast(float, row[POLICY_MARKET_AXIS])
                ),
            }
            for row in result.loadings
        ]
        print(format_table(rounded_loadings))

    def export_report(
        self,
        output_path: str,
        pca_result: NarrativePCAResult | None = None,
        chaos_result: ChaosStabilityResult | None = None,
        sensitivity_result: SensitivityResult | None = None,
        synthetic_validation_result: SyntheticValidationResult | None = None,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Vatican / AI Narrative Bayesian Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Current Posteriors",
            "",
            "These probabilities are subjective model beliefs after applying the evidence ledger. They should be read as calibrated degrees of belief, not proof.",
            "",
            markdown_table(self.posterior_table()),
            "",
            "## How To Read The Probabilities",
            "",
            probability_legend(),
            "",
            "## Evidence Ledger",
            "",
            markdown_table(self.event_table()) if self.events else "No evidence events.",
            "",
            "## Market Signals",
            "",
            markdown_table(self.market_table()) if self.market_signals else "No market signals.",
            "",
            "## Market Opinion Pool",
            "",
            markdown_table(self.market_opinion_pool_table())
            if self.market_signals
            else "No market opinion pool rows.",
            "",
            "## Market Kalman Filter",
            "",
            markdown_table(self.market_kalman_table())
            if self.market_signals
            else "No market Kalman rows.",
            "",
            "## Market Consistency Checks",
            "",
            markdown_table(self.market_arbitrage_table())
            if self.market_signals
            else "No market consistency rows.",
            "",
            "## Posterior History",
            "",
            markdown_table(self.history_table()) if self.posterior_history else "No posterior updates.",
        ]

        if pca_result is not None:
            pca_rows = [
                {
                    "label": label,
                    MORAL_LEGITIMACY_AXIS: round(component[0], 4),
                    "moral_legitimacy_position": moral_legitimacy_score_phrase(component[0]),
                    POLICY_MARKET_AXIS: round(component[1], 4),
                    "policy_market_position": policy_market_score_phrase(component[1]),
                }
                for label, component in zip(pca_result.labels, pca_result.components)
            ]
            loading_rows = [
                {
                    "dimension": row["dimension"],
                    MORAL_LEGITIMACY_AXIS: round(
                        cast(float, row[MORAL_LEGITIMACY_AXIS]), 4
                    ),
                    "moral_legitimacy_read": pca_loading_phrase(
                        cast(float, row[MORAL_LEGITIMACY_AXIS])
                    ),
                    POLICY_MARKET_AXIS: round(cast(float, row[POLICY_MARKET_AXIS]), 4),
                    "policy_market_read": pca_loading_phrase(
                        cast(float, row[POLICY_MARKET_AXIS])
                    ),
                }
                for row in pca_result.loadings
            ]
            lines.extend(
                [
                    "",
                    "## Narrative PCA",
                    "",
                    "Axis scores show where each narrative item sits relative to the center of the PCA map. Larger absolute values mean the item is farther from the average item on that axis. The sign is directional within this run.",
                    "",
                    pca_score_legend(),
                    "",
                    f"Explained variance: {[round(value, 3) for value in pca_result.explained_variance]}",
                    "",
                    markdown_table(pca_rows),
                    "",
                    "## PCA Loadings",
                    "",
                    "Loadings show how strongly each narrative dimension pulls toward the positive or negative pole of a component. The sign is directional within this run, not a moral judgment.",
                    "",
                    pca_loading_legend(),
                    "",
                    markdown_table(loading_rows),
                ]
            )

        if chaos_result is not None:
            lines.extend(
                [
                    "",
                    "## Chaos / Stability Diagnostic",
                    "",
                    "This diagnostic treats the posterior vector as a small discrete dynamical system. It is a local stability read, not a proof of chaos.",
                    "",
                    f"Classification: {chaos_result.classification}",
                    "",
                    f"Confidence: {chaos_result.confidence}",
                    "",
                    f"Lyapunov-style proxy: {_format_optional_float(chaos_result.lyapunov_proxy)}",
                    "",
                    chaos_result.explanation,
                    "",
                    "### Event-to-event Movement",
                    "",
                    markdown_table(chaos_result.step_rows)
                    if chaos_result.step_rows
                    else "No event movement rows.",
                    "",
                    "### Hypothesis-level Local Behavior",
                    "",
                    markdown_table(chaos_result.hypothesis_rows)
                    if chaos_result.hypothesis_rows
                    else "No hypothesis-level rows.",
                    "",
                    "### Recurrence Quantification",
                    "",
                    markdown_table(chaos_result.recurrence_rows)
                    if chaos_result.recurrence_rows
                    else "No recurrence rows.",
                    "",
                    "### Rosenstein Lyapunov Diagnostic",
                    "",
                    markdown_table(chaos_result.lyapunov_rows)
                    if chaos_result.lyapunov_rows
                    else "No Lyapunov rows.",
                    "",
                    "### Correlation Dimension",
                    "",
                    markdown_table(chaos_result.dimension_rows)
                    if chaos_result.dimension_rows
                    else "No correlation dimension rows.",
                    "",
                    "### Surrogate Test",
                    "",
                    markdown_table(chaos_result.surrogate_rows)
                    if chaos_result.surrogate_rows
                    else "No surrogate test rows.",
                ]
            )

        if sensitivity_result is not None:
            lines.extend(
                [
                    "",
                    "## Sensitivity Analysis",
                    "",
                    sensitivity_result.summary,
                    "",
                    "### Posterior Robustness",
                    "",
                    markdown_table(sensitivity_result.rows)
                    if sensitivity_result.rows
                    else "No posterior robustness rows.",
                    "",
                    "### Classification Robustness",
                    "",
                    markdown_table(sensitivity_result.classification_rows)
                    if sensitivity_result.classification_rows
                    else "No classification robustness rows.",
                ]
            )

        if synthetic_validation_result is not None:
            lines.extend(
                [
                    "",
                    "## Synthetic Dynamics Validation",
                    "",
                    synthetic_validation_result.summary,
                    "",
                    markdown_table(synthetic_validation_result.rows)
                    if synthetic_validation_result.rows
                    else "No synthetic validation rows.",
                ]
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved report to {path}")

    @staticmethod
    def _bayes_update(prior: float, likelihood_ratio: float) -> float:
        if not 0 < prior < 1:
            raise ValueError("Prior must be between 0 and 1.")
        if likelihood_ratio <= 0:
            raise ValueError("Likelihood ratio must be positive.")

        prior_odds = prior / (1 - prior)
        posterior_odds = prior_odds * likelihood_ratio
        return posterior_odds / (1 + posterior_odds)

    @staticmethod
    def _temper_likelihood_ratio(likelihood_ratio: float, confidence: float) -> float:
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if likelihood_ratio <= 0:
            raise ValueError("Likelihood ratio must be positive.")
        return float(math.exp(math.log(likelihood_ratio) * confidence))

    def _dependency_adjusted_likelihood_ratio(
        self, likelihood_ratio: float, dependency_factors: Mapping[str, float]
    ) -> float:
        if likelihood_ratio <= 0:
            raise ValueError("Likelihood ratio must be positive.")
        if not dependency_factors:
            return likelihood_ratio

        overlap = sum(
            self.factor_exposure.get(factor, 0.0) * max(strength, 0.0)
            for factor, strength in dependency_factors.items()
        )
        independence_weight = 1 / (1 + overlap)
        return float(math.exp(math.log(likelihood_ratio) * independence_weight))

    @staticmethod
    def _parse_volume(volume: str | float | int | None) -> float | None:
        if volume is None:
            return None
        if isinstance(volume, (float, int)):
            return float(volume)

        cleaned = volume.strip().replace("$", "").replace(",", "").upper()
        if cleaned in {"", "N/A", "NA", "NONE"}:
            return None
        multiplier = 1.0
        if cleaned.endswith("K"):
            multiplier = 1_000.0
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1_000_000.0
            cleaned = cleaned[:-1]
        return float(cleaned) * multiplier


def format_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "(empty)"

    columns = list(rows[0].keys())
    rendered_rows = [[_format_cell(row.get(column, "")) for column in columns] for row in rows]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rendered_rows))
        for index, column in enumerate(columns)
    ]
    header = "  ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in rendered_rows
    ]
    return "\n".join([header, divider, *body])


def markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""

    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_escape_markdown_cell(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def print_console_note(title: str, message: str) -> None:
    print(f"  {title}: {message}")


def clamp_probability(value: float) -> float:
    return min(max(value, 1e-6), 1 - 1e-6)


def clamp_unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def logit(probability: float) -> float:
    bounded = clamp_probability(probability)
    return math.log(bounded / (1 - bounded))


def inverse_logit(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def evidence_weight(event: EvidenceEvent) -> float:
    quality_weights = {
        "primary": 1.4,
        "secondary": 1.0,
        "market": 0.9,
        "hypothetical": 0.55,
        "unverified": 0.35,
    }
    return event.confidence * quality_weights[event.quality]


def scale_likelihood_ratio(likelihood_ratio: float, scale: float) -> float:
    if likelihood_ratio <= 0:
        raise ValueError("Likelihood ratio must be positive.")
    return math.exp(math.log(likelihood_ratio) * scale)


def uncertainty_phrase(interval_width: float) -> str:
    if interval_width < 0.12:
        return "tight"
    if interval_width < 0.25:
        return "moderate"
    if interval_width < 0.40:
        return "wide"
    return "very wide"


def format_factor_map(values: Mapping[str, float]) -> str:
    if not values:
        return "N/A"
    return ", ".join(f"{key}:{value:.2f}" for key, value in sorted(values.items()))


def default_narrative_matrix() -> tuple[list[str], list[str], list[list[float]]]:
    dimensions = [
        "institutional_authority",
        "ai_governance_relevance",
        "media_amplification",
        "policy_coupling",
        "anthropic_brand_benefit",
        "religious_symbolic_power",
        "market_actionability",
    ]
    labels = [
        "May25_Event",
        "Indulgences_Parallel",
        "Conjunction",
        "Self_Steering",
        "Regulatory_Followthrough",
        "Market_Repricing",
    ]
    matrix = [
        [0.8, 0.9, 0.4, 0.7, 0.85, 0.9, 0.75],
        [0.7, 0.8, 0.5, 0.8, 0.8, 0.85, 0.8],
        [0.6, 0.7, 0.3, 0.9, 0.9, 0.95, 0.85],
        [0.9, 0.6, 0.2, 0.4, 0.7, 0.8, 0.9],
        [0.55, 0.95, 0.65, 0.95, 0.45, 0.6, 0.7],
        [0.3, 0.55, 0.75, 0.35, 0.8, 0.35, 0.95],
    ]
    return dimensions, labels, matrix


def group_market_signals(signals: Sequence[MarketSignal]) -> dict[str, list[MarketSignal]]:
    grouped: dict[str, list[MarketSignal]] = {}
    for signal in signals:
        grouped.setdefault(signal.question, []).append(signal)
    return grouped


def kalman_filter_market_odds(
    signals: Sequence[MarketSignal], config: MarketFilterConfig
) -> list[dict[str, float]]:
    if not signals:
        return []
    state = clamp_probability(signals[0].yes_odds)
    variance = config.initial_variance
    estimates = []
    for signal in signals:
        observation = clamp_probability(signal.yes_odds)
        variance += config.process_noise
        observation_variance = config.observation_noise_floor / max(
            signal.liquidity_weight, 0.05
        )
        kalman_gain = variance / (variance + observation_variance)
        state = clamp_probability(state + kalman_gain * (observation - state))
        variance = (1 - kalman_gain) * variance
        estimates.append(
            {
                "state": state,
                "variance": variance,
                "kalman_gain": kalman_gain,
            }
        )
    return estimates


def euclidean_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def flatten_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [value for vector in vectors for value in vector]


def vector_angle(left: Sequence[float] | None, right: Sequence[float]) -> float | None:
    if left is None:
        return None
    left_norm = euclidean_norm(list(left))
    right_norm = euclidean_norm(list(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    dot = sum(left[index] * right[index] for index in range(min(len(left), len(right))))
    cosine = max(min(dot / (left_norm * right_norm), 1.0), -1.0)
    return math.degrees(math.acos(cosine))


def temporal_pca_phrase(rotation: float | None) -> str:
    if rotation is None:
        return "baseline window"
    if rotation < 10:
        return "stable narrative axis"
    if rotation < 35:
        return "moderate narrative rotation"
    return "major narrative axis rotation"


def takens_embedding_vectors(
    state_vectors: list[list[float]], embedding_dimension: int, delay: int
) -> list[list[float]]:
    if embedding_dimension < 1:
        raise ValueError("Embedding dimension must be at least 1.")
    if delay < 1:
        raise ValueError("Delay must be at least 1.")
    max_start = len(state_vectors) - (embedding_dimension - 1) * delay
    if max_start <= 0:
        return []
    return [
        flatten_vectors(
            [state_vectors[start + offset * delay] for offset in range(embedding_dimension)]
        )
        for start in range(max_start)
    ]


def recurrence_quantification(
    embedded_points: list[list[float]], threshold: float
) -> dict[str, float]:
    point_count = len(embedded_points)
    if point_count < 2:
        return {"recurrence_rate": 0.0, "determinism": 0.0, "laminarity": 0.0}

    matrix = recurrence_matrix(embedded_points, threshold)
    recurrence_count = sum(sum(row) for row in matrix)
    possible = point_count * (point_count - 1)
    recurrence_rate = recurrence_count / possible if possible else 0.0
    diagonal_points = count_line_points(matrix, diagonal=True, minimum_length=2)
    vertical_points = count_line_points(matrix, diagonal=False, minimum_length=2)
    determinism = diagonal_points / recurrence_count if recurrence_count else 0.0
    laminarity = vertical_points / recurrence_count if recurrence_count else 0.0
    return {
        "recurrence_rate": recurrence_rate,
        "determinism": determinism,
        "laminarity": laminarity,
    }


def recurrence_matrix(embedded_points: list[list[float]], threshold: float) -> list[list[int]]:
    point_count = len(embedded_points)
    return [
        [
            1
            if row != column
            and vector_distance(embedded_points[row], embedded_points[column]) <= threshold
            else 0
            for column in range(point_count)
        ]
        for row in range(point_count)
    ]


def vector_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return euclidean_norm([left[index] - right[index] for index in range(len(left))])


def nearest_neighbor_indices(
    embedded_points: list[list[float]], theiler_window: int
) -> list[int | None]:
    neighbors: list[int | None] = []
    for index, point in enumerate(embedded_points):
        best_index = None
        best_distance = math.inf
        for candidate_index, candidate in enumerate(embedded_points):
            if abs(candidate_index - index) <= theiler_window:
                continue
            distance = vector_distance(point, candidate)
            if 0 < distance < best_distance:
                best_distance = distance
                best_index = candidate_index
        neighbors.append(best_index)
    return neighbors


def rosenstein_mle(
    embedded_points: list[list[float]],
    theiler_window: int,
    max_horizon: int,
) -> tuple[float | None, list[dict[str, object]]]:
    if len(embedded_points) < 3:
        return None, []

    neighbors = nearest_neighbor_indices(embedded_points, theiler_window)
    divergence_rows = []
    log_divergences = []
    horizon_limit = min(max_horizon, len(embedded_points) - 2)
    for horizon in range(1, horizon_limit + 1):
        logs = []
        for index, neighbor_index in enumerate(neighbors):
            if neighbor_index is None:
                continue
            if index + horizon >= len(embedded_points):
                continue
            if neighbor_index + horizon >= len(embedded_points):
                continue
            distance = vector_distance(
                embedded_points[index + horizon],
                embedded_points[neighbor_index + horizon],
            )
            if distance > 1e-12:
                logs.append(math.log(distance))
        if logs:
            average_log_distance = sum(logs) / len(logs)
            log_divergences.append((float(horizon), average_log_distance))
            divergence_rows.append(
                {
                    "horizon": horizon,
                    "avg_log_divergence": round(average_log_distance, 4),
                    "pairs": len(logs),
                }
            )

    if len(log_divergences) < 2:
        return None, divergence_rows
    return linear_regression_slope(log_divergences), divergence_rows


def lyapunov_diagnostic_rows(
    embedded_points: list[list[float]], config: DynamicalAnalysisConfig
) -> list[dict[str, object]]:
    if len(embedded_points) < config.min_points_for_mle:
        return [
            {
                "metric": "rosenstein_mle",
                "value": "insufficient points",
                "read": (
                    f"need at least {config.min_points_for_mle} embedded points; "
                    f"have {len(embedded_points)}"
                ),
            }
        ]
    mle, divergence_rows = rosenstein_mle(
        embedded_points, config.theiler_window, config.max_lyapunov_horizon
    )
    if mle is None:
        return [
            {
                "metric": "rosenstein_mle",
                "value": "not estimable",
                "read": "nearest-neighbor divergence curve is too sparse",
            }
        ]
    rows = [
        {
            "metric": "rosenstein_mle",
            "value": round(mle, 5),
            "read": lyapunov_phrase(mle),
        }
    ]
    rows.extend(
        {
            "metric": f"divergence_t{row['horizon']}",
            "value": row["avg_log_divergence"],
            "read": f"{row['pairs']} neighbor pairs",
        }
        for row in divergence_rows
    )
    return rows


def correlation_dimension_rows(
    embedded_points: list[list[float]], config: DynamicalAnalysisConfig
) -> list[dict[str, object]]:
    if len(embedded_points) < config.min_points_for_correlation_dimension:
        return [
            {
                "metric": "correlation_dimension_d2",
                "value": "insufficient points",
                "read": (
                    "correlation dimension needs a longer trajectory; "
                    f"have {len(embedded_points)} embedded points"
                ),
            }
        ]
    distances = pairwise_distances(embedded_points)
    positive_distances = [distance for distance in distances if distance > 1e-12]
    if len(positive_distances) < 4:
        return [
            {
                "metric": "correlation_dimension_d2",
                "value": "not estimable",
                "read": "not enough nonzero pairwise distances",
            }
        ]
    radii = geometric_radii(min(positive_distances), max(positive_distances), 8)
    points = []
    pair_count = len(positive_distances)
    for radius in radii:
        correlation_integral = (
            sum(1 for distance in positive_distances if distance <= radius) / pair_count
        )
        if 0 < correlation_integral < 1:
            points.append((math.log(radius), math.log(correlation_integral)))
    if len(points) < 3:
        return [
            {
                "metric": "correlation_dimension_d2",
                "value": "not estimable",
                "read": "no stable scaling region was found",
            }
        ]
    slope = linear_regression_slope(points)
    return [
        {
            "metric": "correlation_dimension_d2",
            "value": round(slope, 4),
            "read": correlation_dimension_phrase(slope),
        },
        {
            "metric": "scaling_points",
            "value": len(points),
            "read": "log-log points used for slope",
        },
    ]


def surrogate_lyapunov_rows(
    embedded_points: list[list[float]],
    config: DynamicalAnalysisConfig,
    observed_mle: float | None,
) -> list[dict[str, object]]:
    if observed_mle is None:
        return [
            {
                "metric": "surrogate_test",
                "value": "skipped",
                "read": "observed MLE is unavailable",
            }
        ]
    if len(embedded_points) < config.min_points_for_mle:
        return [
            {
                "metric": "surrogate_test",
                "value": "insufficient points",
                "read": "need a longer embedded trajectory",
            }
        ]
    surrogate_mles = []
    for run_index in range(config.surrogate_runs):
        surrogate = deterministic_surrogate_shuffle(
            embedded_points, config.surrogate_seed + run_index
        )
        mle, _ = rosenstein_mle(
            surrogate, config.theiler_window, config.max_lyapunov_horizon
        )
        if mle is not None:
            surrogate_mles.append(mle)
    if not surrogate_mles:
        return [
            {
                "metric": "surrogate_test",
                "value": "not estimable",
                "read": "surrogate MLE curves were too sparse",
            }
        ]
    surrogate_mean = sum(surrogate_mles) / len(surrogate_mles)
    surrogate_std = sample_std(surrogate_mles)
    z_score = (
        (observed_mle - surrogate_mean) / surrogate_std
        if surrogate_std > 1e-12
        else 0.0
    )
    return [
        {
            "metric": "surrogate_mean_mle",
            "value": round(surrogate_mean, 5),
            "read": "deterministic shuffle surrogate baseline",
        },
        {
            "metric": "surrogate_z_score",
            "value": round(z_score, 4),
            "read": surrogate_test_phrase(z_score),
        },
    ]


def count_line_points(
    matrix: list[list[int]], diagonal: bool, minimum_length: int
) -> int:
    size = len(matrix)
    total = 0
    if diagonal:
        offsets = range(-(size - 1), size)
        for offset in offsets:
            run = 0
            for row in range(size):
                column = row + offset
                if 0 <= column < size and matrix[row][column]:
                    run += 1
                else:
                    if run >= minimum_length:
                        total += run
                    run = 0
            if run >= minimum_length:
                total += run
    else:
        for column in range(size):
            run = 0
            for row in range(size):
                if matrix[row][column]:
                    run += 1
                else:
                    if run >= minimum_length:
                        total += run
                    run = 0
            if run >= minimum_length:
                total += run
    return total


def pairwise_distances(points: list[list[float]]) -> list[float]:
    distances = []
    for left_index in range(len(points)):
        for right_index in range(left_index + 1, len(points)):
            distances.append(vector_distance(points[left_index], points[right_index]))
    return distances


def geometric_radii(min_radius: float, max_radius: float, count: int) -> list[float]:
    if count <= 1:
        return [max(min_radius, 1e-12)]
    low = math.log(max(min_radius, 1e-12))
    high = math.log(max(max_radius, min_radius * 1.01, 1e-12))
    return [math.exp(low + (high - low) * index / (count - 1)) for index in range(count)]


def linear_regression_slope(points: Sequence[tuple[float, float]]) -> float:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def deterministic_surrogate_shuffle(
    points: list[list[float]], seed: int
) -> list[list[float]]:
    keyed_points = [
        (math.sin((index + 1) * (seed + 1) * 12.9898) % 1.0, point)
        for index, point in enumerate(points)
    ]
    return [point[:] for _, point in sorted(keyed_points, key=lambda item: item[0])]


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def extract_metric_value(
    rows: Sequence[Mapping[str, object]], metric_name: str
) -> float | None:
    for row in rows:
        if row.get("metric") != metric_name:
            continue
        value = row.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def rqa_metric_phrase(metric: str, value: float) -> str:
    if metric == "recurrence_rate":
        if value < 0.05:
            return "little repeated state structure"
        if value < 0.20:
            return "some repeated state structure"
        return "strong repeated state structure"
    if metric == "determinism":
        if value < 0.35:
            return "weak sequential patterning"
        if value < 0.70:
            return "moderate sequential patterning"
        return "strong sequential patterning"
    if metric == "laminarity":
        if value < 0.35:
            return "few plateau-like regimes"
        if value < 0.70:
            return "some plateau-like regimes"
        return "strong plateau-like regimes"
    return "diagnostic metric"


def lyapunov_phrase(mle: float) -> str:
    if mle > 0.03:
        return "positive MLE; locally divergent / chaos-compatible"
    if mle < -0.03:
        return "negative MLE; locally convergent"
    return "near-zero MLE; periodic or marginal dynamics compatible"


def correlation_dimension_phrase(dimension: float) -> str:
    if dimension < 1.2:
        return "very low-dimensional attractor estimate"
    if dimension < 3.5:
        return "low-dimensional structure estimate"
    return "high-dimensional or noise-dominated estimate"


def surrogate_test_phrase(z_score: float) -> str:
    if z_score > 2.0:
        return "observed divergence exceeds surrogate baseline"
    if z_score < -2.0:
        return "observed divergence is below surrogate baseline"
    return "not clearly separated from surrogate baseline"


def local_axis_growth_rate(series: list[float]) -> float | None:
    magnitudes = [abs(value) for value in series if abs(value) > 1e-9]
    if len(magnitudes) < 2:
        return None
    return magnitudes[-1] / magnitudes[-2]


def stability_step_phrase(growth_ratio: float | None) -> str:
    if growth_ratio is None:
        return "baseline movement"
    if growth_ratio < 0.70:
        return "movement is contracting"
    if growth_ratio <= 1.15:
        return "movement is roughly steady"
    return "movement is expanding"


def local_stability_phrase(local_growth: float | None) -> str:
    if local_growth is None:
        return "too little movement to classify"
    if local_growth < 0.70:
        return "locally contracting"
    if local_growth <= 1.15:
        return "locally steady"
    return "locally expanding"


def delta_direction_phrase(series: list[float]) -> str:
    if not series:
        return "no movement"
    positive = any(value > 1e-9 for value in series)
    negative = any(value < -1e-9 for value in series)
    if positive and negative:
        return "direction changed / oscillatory"
    if positive:
        return "moving upward"
    if negative:
        return "moving downward"
    return "flat"


def classify_stability(
    step_norms: list[float],
    local_rates: list[float],
    lyapunov_proxy: float | None,
) -> str:
    if not step_norms:
        return "insufficient trajectory"

    contracting_axes = sum(1 for rate in local_rates if rate < 0.85)
    expanding_axes = sum(1 for rate in local_rates if rate > 1.15)
    has_mixed_axes = contracting_axes > 0 and expanding_axes > 0
    final_is_smaller = step_norms[-1] < step_norms[0]
    final_is_larger = step_norms[-1] > step_norms[0]

    if has_mixed_axes:
        return "saddle-like mixed stability"
    if lyapunov_proxy is not None and lyapunov_proxy < -0.15 and final_is_smaller:
        return "stable / converging attractor-like path"
    if lyapunov_proxy is not None and lyapunov_proxy > 0.15 and final_is_larger:
        return "unstable / diverging path"
    if all(rate <= 1.15 for rate in local_rates) and final_is_smaller:
        return "weakly stable path"
    if all(rate >= 0.85 for rate in local_rates) and final_is_larger:
        return "weakly unstable path"
    return "indeterminate / near transition"


def chaos_confidence_phrase(step_count: int, ratios: list[float]) -> str:
    if step_count < 3:
        return "low: very short event trajectory"
    if len(ratios) < 3:
        return "medium-low: enough for a local read, not enough for robust chaos claims"
    return "medium: local trajectory has several movements, still not a full dynamical model"


def chaos_classification_explanation(
    classification: str, lyapunov_proxy: float | None
) -> str:
    if lyapunov_proxy is None:
        lyapunov_text = "No Lyapunov-style proxy could be estimated."
    elif lyapunov_proxy < 0:
        lyapunov_text = "The Lyapunov-style proxy is negative, so recent movements shrink on average."
    elif lyapunov_proxy > 0:
        lyapunov_text = "The Lyapunov-style proxy is positive, so recent movements grow on average."
    else:
        lyapunov_text = "The Lyapunov-style proxy is near zero, so movement size is roughly balanced."

    explanations = {
        "stable / converging attractor-like path": (
            "The posterior vector is moving in smaller steps, which resembles convergence "
            "toward a local attractor."
        ),
        "weakly stable path": (
            "The posterior vector is not fully settled, but the step size is shrinking enough "
            "to look weakly stabilizing."
        ),
        "unstable / diverging path": (
            "The posterior vector is taking larger steps over time, which resembles local divergence."
        ),
        "weakly unstable path": (
            "The posterior vector is drifting with growing movement, but not sharply enough "
            "to call it strongly unstable."
        ),
        "saddle-like mixed stability": (
            "Some hypothesis directions are contracting while others are expanding. That mixed "
            "geometry is the hallmark of a saddle-like region."
        ),
        "indeterminate / near transition": (
            "The trajectory has mixed or weak signals, so it may be near a transition boundary."
        ),
    }
    return f"{explanations.get(classification, 'The trajectory is too short to classify.')} {lyapunov_text}"


def sensitivity_spread_phrase(spread: float) -> str:
    if spread < 0.05:
        return "high"
    if spread < 0.12:
        return "moderate"
    if spread < 0.25:
        return "fragile"
    return "very fragile"


def sensitivity_summary(
    posterior_rows: Sequence[Mapping[str, object]],
    classification_rows: Sequence[Mapping[str, object]],
) -> str:
    if not posterior_rows:
        return "No sensitivity runs were completed."

    worst_spread = max(cast(float, row["spread"]) for row in posterior_rows)
    posterior_read = sensitivity_spread_phrase(worst_spread)
    if classification_rows:
        top_classification = classification_rows[0]["classification"]
        top_share = cast(float, classification_rows[0]["share"])
        classification_text = (
            f"Dominant stability classification is '{top_classification}' "
            f"in {top_share:.1%} of perturbation runs."
        )
    else:
        classification_text = "No stability classifications were generated."

    return (
        f"Posterior robustness is {posterior_read}; worst posterior spread is "
        f"{worst_spread:.3f}. {classification_text}"
    )


def run_synthetic_dynamics_validation() -> SyntheticValidationResult:
    cases = [
        ("logistic_stable", logistic_series(rate=2.8, initial=0.21, steps=80), "stable"),
        ("logistic_chaotic", logistic_series(rate=3.9, initial=0.21, steps=80), "chaotic"),
        ("random_walk", bounded_random_walk(seed=23, steps=80), "stochastic"),
    ]
    rows = []
    for name, series, expected in cases:
        state_vectors = [[value] for value in series]
        embedded = takens_embedding_vectors(state_vectors, embedding_dimension=3, delay=1)
        config = DynamicalAnalysisConfig(
            embedding_dimension=3,
            delay=1,
            min_points_for_mle=12,
            min_points_for_correlation_dimension=12,
            max_lyapunov_horizon=8,
        )
        mle_rows = lyapunov_diagnostic_rows(embedded, config)
        mle = extract_metric_value(mle_rows, "rosenstein_mle")
        predicted = synthetic_classification_from_mle(mle)
        rows.append(
            {
                "case": name,
                "expected": expected,
                "predicted": predicted,
                "mle": round(mle, 5) if mle is not None else "N/A",
                "match": predicted == expected
                or (expected == "stochastic" and predicted in {"chaotic", "indeterminate"}),
            }
        )
    matches = sum(1 for row in rows if row["match"])
    summary = f"Synthetic dynamics validation matched {matches}/{len(rows)} coarse labels."
    return SyntheticValidationResult(rows=rows, summary=summary)


def print_synthetic_validation_result(result: SyntheticValidationResult) -> None:
    print("\n=== Synthetic Dynamics Validation ===")
    print_console_note(
        "What this checks",
        "The same Lyapunov machinery is tested on toy systems with known behavior. This is not "
        "validation of the papal narrative; it validates that the diagnostic reacts sensibly.",
    )
    print(result.summary)
    print(format_table(result.rows))


def logistic_series(rate: float, initial: float, steps: int) -> list[float]:
    values = [clamp_unit_interval(initial)]
    for _ in range(steps - 1):
        values.append(rate * values[-1] * (1 - values[-1]))
    return values


def bounded_random_walk(seed: int, steps: int) -> list[float]:
    value = 0.5
    values = [value]
    for index in range(steps - 1):
        shock = ((math.sin((seed + 1) * (index + 1) * 78.233) % 1.0) - 0.5) * 0.16
        value = clamp_unit_interval(value + shock)
        values.append(value)
    return values


def synthetic_classification_from_mle(mle: float | None) -> str:
    if mle is None:
        return "indeterminate"
    if mle > 0.03:
        return "chaotic"
    if mle < -0.03:
        return "stable"
    return "indeterminate"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def probability_phrase(probability: float) -> str:
    if probability < 0.05:
        return "effectively ruled out"
    if probability < 0.20:
        return "unlikely"
    if probability < 0.40:
        return "possible but not favored"
    if probability < 0.60:
        return "uncertain / roughly balanced"
    if probability < 0.75:
        return "more likely than not"
    if probability < 0.90:
        return "strongly favored"
    if probability < 0.97:
        return "very strongly favored"
    return "near certain within this model"


def market_signal_phrase(weighted_signal: float) -> str:
    if weighted_signal <= -0.15:
        return "meaningful negative market pressure"
    if weighted_signal <= -0.05:
        return "modest negative market pressure"
    if weighted_signal < 0.05:
        return "weak or neutral market pressure"
    if weighted_signal < 0.15:
        return "modest positive market pressure"
    return "meaningful positive market pressure"


def liquidity_phrase(liquidity_weight: float) -> str:
    if liquidity_weight < 0.30:
        return "thin or unknown liquidity"
    if liquidity_weight < 0.60:
        return "moderate liquidity"
    if liquidity_weight < 0.85:
        return "good liquidity"
    return "high liquidity"


def probability_legend() -> str:
    rows = [
        {"range": "0-5%", "plain_language": "effectively ruled out"},
        {"range": "5-20%", "plain_language": "unlikely"},
        {"range": "20-40%", "plain_language": "possible but not favored"},
        {"range": "40-60%", "plain_language": "uncertain / roughly balanced"},
        {"range": "60-75%", "plain_language": "more likely than not"},
        {"range": "75-90%", "plain_language": "strongly favored"},
        {"range": "90-97%", "plain_language": "very strongly favored"},
        {"range": "97-100%", "plain_language": "near certain within this model"},
    ]
    return markdown_table(rows)


def pca_loading_phrase(loading: float) -> str:
    magnitude = abs(loading)
    if magnitude < 0.15:
        strength = "minimal pull"
    elif magnitude < 0.35:
        strength = "weak pull"
    elif magnitude < 0.55:
        strength = "moderate pull"
    elif magnitude < 0.75:
        strength = "strong pull"
    else:
        strength = "dominant pull"

    if loading > 0:
        return f"{strength} toward positive pole"
    if loading < 0:
        return f"{strength} toward negative pole"
    return "no directional pull"


def pca_axis_score_phrase(score: float, positive_pole: str, negative_pole: str) -> str:
    magnitude = abs(score)
    if magnitude < 0.25:
        distance = "near the center"
    elif magnitude < 0.75:
        distance = "slightly offset"
    elif magnitude < 1.50:
        distance = "clearly offset"
    elif magnitude < 2.50:
        distance = "strongly offset"
    else:
        distance = "extreme outlier"

    if score > 0:
        return f"{distance} toward {positive_pole}"
    if score < 0:
        return f"{distance} toward {negative_pole}"
    return "exactly centered on this axis"


def moral_legitimacy_score_phrase(score: float) -> str:
    return pca_axis_score_phrase(
        score,
        positive_pole="moral legitimacy framing",
        negative_pole="less moral-legitimacy framing",
    )


def policy_market_score_phrase(score: float) -> str:
    return pca_axis_score_phrase(
        score,
        positive_pole="policy / market action framing",
        negative_pole="less policy / market action framing",
    )


def pca_score_legend() -> str:
    rows = [
        {"absolute_score": "0.00-0.25", "plain_language": "near the center"},
        {"absolute_score": "0.25-0.75", "plain_language": "slightly offset"},
        {"absolute_score": "0.75-1.50", "plain_language": "clearly offset"},
        {"absolute_score": "1.50-2.50", "plain_language": "strongly offset"},
        {"absolute_score": "2.50+", "plain_language": "extreme outlier"},
    ]
    return markdown_table(rows)


def pca_loading_legend() -> str:
    rows = [
        {"absolute_loading": "0.00-0.15", "plain_language": "minimal pull"},
        {"absolute_loading": "0.15-0.35", "plain_language": "weak pull"},
        {"absolute_loading": "0.35-0.55", "plain_language": "moderate pull"},
        {"absolute_loading": "0.55-0.75", "plain_language": "strong pull"},
        {"absolute_loading": "0.75-1.00", "plain_language": "dominant pull"},
    ]
    return markdown_table(rows)


def _format_cell(value: object) -> str:
    if value is None:
        return "N/A"
    return str(value)


def _escape_markdown_cell(value: object) -> str:
    return _format_cell(value).replace("|", "\\|").replace("\n", " ")


def build_demo_tracker() -> ResonanceBayesianTracker:
    tracker = ResonanceBayesianTracker()

    tracker.add_event(
        EvidenceEvent.from_date(
            date_str="2026-05-25",
            title="Vatican + AI lab public event with humility framing",
            quality="hypothetical",
            source="manual scenario",
            rationale="Public institutional proximity would strengthen symbolic legitimation and brand effects more than direct policy impact.",
            confidence=0.72,
            likelihood_ratios={
                "governance_influence": 1.45,
                "symbolic_legitimation": 2.80,
                "anthropic_brand_benefit": 2.10,
                "low_policy_impact": 0.82,
            },
            dependency_factors={
                "institutional_signal": 0.75,
                "media_attention": 0.45,
            },
            narrative_features={
                "institutional_authority": 0.80,
                "ai_governance_relevance": 0.90,
                "media_amplification": 0.40,
                "policy_coupling": 0.70,
                "anthropic_brand_benefit": 0.85,
                "religious_symbolic_power": 0.90,
                "market_actionability": 0.75,
            },
        )
    )
    tracker.add_event(
        EvidenceEvent.from_date(
            date_str="2026-05-26",
            title="Media framing emphasizes Church guidance of responsible AI",
            quality="secondary",
            source="manual media summary",
            rationale="Media uptake matters most for symbolic legitimacy; it is weaker evidence for direct governance change.",
            confidence=0.65,
            likelihood_ratios={
                "governance_influence": 1.20,
                "symbolic_legitimation": 1.90,
                "anthropic_brand_benefit": 1.65,
                "low_policy_impact": 0.95,
            },
            dependency_factors={
                "media_attention": 0.85,
                "institutional_signal": 0.25,
            },
            narrative_features={
                "institutional_authority": 0.70,
                "ai_governance_relevance": 0.80,
                "media_amplification": 0.65,
                "policy_coupling": 0.62,
                "anthropic_brand_benefit": 0.82,
                "religious_symbolic_power": 0.78,
                "market_actionability": 0.60,
            },
        )
    )
    tracker.add_event(
        EvidenceEvent.from_date(
            date_str="2026-05-27",
            title="AI lab emphasizes ongoing discernment partnership",
            quality="hypothetical",
            source="manual scenario",
            rationale="Partnership language increases odds of sustained narrative value, but still needs policy follow-through.",
            confidence=0.68,
            likelihood_ratios={
                "governance_influence": 1.55,
                "symbolic_legitimation": 2.30,
                "anthropic_brand_benefit": 1.85,
                "low_policy_impact": 0.88,
            },
            dependency_factors={
                "partnership_language": 0.80,
                "media_attention": 0.35,
            },
            narrative_features={
                "institutional_authority": 0.68,
                "ai_governance_relevance": 0.88,
                "media_amplification": 0.50,
                "policy_coupling": 0.76,
                "anthropic_brand_benefit": 0.86,
                "religious_symbolic_power": 0.84,
                "market_actionability": 0.66,
            },
        )
    )

    tracker.add_polymarket_bet(
        "2026-05-26",
        "Will US pass major AI safety bill before 2027?",
        0.35,
        "$98k",
        relevance={"governance_influence": 0.55, "low_policy_impact": 0.45},
    )
    tracker.add_polymarket_bet(
        "2026-05-26",
        "Will Vatican issue further AI regulation guidance by end 2026?",
        0.62,
        None,
        relevance={"governance_influence": 0.75, "symbolic_legitimation": 0.45},
    )
    tracker.add_polymarket_bet(
        "2026-05-26",
        "Will Anthropic valuation exceed $500B by Dec 2027?",
        0.48,
        None,
        relevance={"anthropic_brand_benefit": 0.35},
    )

    return tracker


if __name__ == "__main__":
    tracker = build_demo_tracker()
    tracker.show_polymarket_summary()
    tracker.run_chaos_stability_analysis()
    tracker.run_sensitivity_analysis()
    pca = tracker.run_narrative_pca()
    tracker.run_temporal_pca()
    print_synthetic_validation_result(run_synthetic_dynamics_validation())
    tracker.plot_probability_timeline()

