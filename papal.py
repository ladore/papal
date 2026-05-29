from __future__ import annotations

from dataclasses import dataclass
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
    ) -> "EvidenceEvent":
        return cls(
            date=datetime.strptime(date_str, "%Y-%m-%d"),
            title=title,
            quality=quality,
            source=source,
            rationale=rationale,
            likelihood_ratios=likelihood_ratios,
            confidence=confidence,
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


class ResonanceBayesianTracker:
    """
    A single-file evidence ledger for updating multiple narrative hypotheses.

    The model is intentionally modest: it does not prove causality. It keeps a
    disciplined, inspectable trail of priors, evidence, likelihood ratios,
    market signals, and narrative geometry.
    """

    def __init__(self, hypotheses: list[Hypothesis] | None = None):
        self.hypotheses = hypotheses or self.default_hypotheses()
        self.posteriors = {h.key: h.prior for h in self.hypotheses}
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
            weighted_lr = self._temper_likelihood_ratio(raw_lr, event.confidence)
            self.posteriors[hypothesis.key] = self._bayes_update(
                self.posteriors[hypothesis.key], weighted_lr
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

        print(f"[{event.date.date()}] {event.title}")
        print(format_table(self.posterior_table()))

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
            f"liquidity weight: {signal.liquidity_weight:.2f}"
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
            rows.append(
                {
                    "hypothesis": hypothesis.key,
                    "posterior": round(self.posteriors[hypothesis.key], 4),
                    "interpretation": probability_phrase(self.posteriors[hypothesis.key]),
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
        print(format_table(self.market_table()))

    def run_chaos_stability_analysis(self) -> ChaosStabilityResult:
        snapshots = self._posterior_snapshots()
        if len(snapshots) < 3:
            result = ChaosStabilityResult(
                classification="insufficient trajectory",
                confidence="low",
                lyapunov_proxy=None,
                step_rows=[],
                hypothesis_rows=[],
                explanation=(
                    "Need at least two event-to-event movements after the initial prior "
                    "to say anything useful about stability."
                ),
            )
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
        explanation = chaos_classification_explanation(classification, lyapunov_proxy)
        result = ChaosStabilityResult(
            classification=classification,
            confidence=confidence,
            lyapunov_proxy=lyapunov_proxy,
            step_rows=step_rows,
            hypothesis_rows=hypothesis_rows,
            explanation=explanation,
        )
        self.print_chaos_stability_result(result)
        return result

    def print_chaos_stability_result(self, result: ChaosStabilityResult) -> None:
        print("\n=== Chaos / Stability Diagnostic ===")
        print(f"Classification: {result.classification}")
        print(f"Confidence: {result.confidence}")
        if result.lyapunov_proxy is not None:
            print(f"Lyapunov-style proxy: {result.lyapunov_proxy:.4f}")
        print(result.explanation)
        if result.step_rows:
            print("\nEvent-to-event movement:")
            print(format_table(result.step_rows))
        if result.hypothesis_rows:
            print("\nHypothesis-level local behavior:")
            print(format_table(result.hypothesis_rows))

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

    def run_narrative_pca(self) -> NarrativePCAResult | None:
        if np is None or PCA is None or StandardScaler is None:
            print("Skipping narrative PCA because numpy or scikit-learn is not installed.")
            return None

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
        data = np.array(
            [
                [0.8, 0.9, 0.4, 0.7, 0.85, 0.9, 0.75],
                [0.7, 0.8, 0.5, 0.8, 0.8, 0.85, 0.8],
                [0.6, 0.7, 0.3, 0.9, 0.9, 0.95, 0.85],
                [0.9, 0.6, 0.2, 0.4, 0.7, 0.8, 0.9],
                [0.55, 0.95, 0.65, 0.95, 0.45, 0.6, 0.7],
                [0.3, 0.55, 0.75, 0.35, 0.8, 0.35, 0.95],
            ]
        )

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

    def print_pca_result(self, result: NarrativePCAResult) -> None:
        print("\n=== Narrative PCA ===")
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


def euclidean_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


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
    pca = tracker.run_narrative_pca()
    tracker.plot_probability_timeline()

