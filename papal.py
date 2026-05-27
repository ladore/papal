import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ResonanceBayesianTracker:
    """
    Tracks Bayesian updates + Polymarket-related bets on Vatican-Anthropic narrative.
    """
    
    def __init__(self):
        self.timeline = []
        self.posterior_history = []
        self.prior = 0.70  # Initial P(perturbation/normalization dominant)
        self.polymarket_bets = []  # List of (date, market_question, current_odds_yes, volume)
        
    def add_event(self, date_str: str, description: str, likelihood_ratio: float):
        date = datetime.strptime(date_str, "%Y-%m-%d")
        self.timeline.append((date, description, likelihood_ratio))
        
        prior_odds = self.prior / (1 - self.prior)
        posterior_odds = prior_odds * likelihood_ratio
        self.prior = posterior_odds / (1 + posterior_odds)
        
        self.posterior_history.append((date, round(self.prior, 4), description))
        print(f"[{date.date()}] {description[:70]}... → Posterior: {self.prior:.4f}")
    
    def add_polymarket_bet(self, date_str: str, question: str, yes_odds: float, volume: str):
        """Add real or hypothetical Polymarket bet"""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        self.polymarket_bets.append((date, question, yes_odds, volume))
        print(f"Polymarket: '{question}' → Yes odds: {yes_odds:.1%} | Vol: {volume}")
    
    def plot_probability_timeline(self):
        dates = [t[0] for t in self.posterior_history]
        probs = [t[1] for t in self.posterior_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, probs, marker='o', linewidth=2, label="P(Perturbation Dominant)")
        plt.title("Bayesian Probability Timeline: Vatican-Anthropic Conjunction")
        plt.ylabel("Probability")
        plt.xlabel("Date")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def show_polymarket_summary(self):
        print("\n=== Relevant Polymarket Bets ===")
        for bet in self.polymarket_bets:
            print(f"{bet[0].date()}: {bet[1]} | Yes: {bet[2]:.1%} | Vol: {bet[3]}")
    
    def run_narrative_pca(self):
        # Same PCA as before (narrative dimensions)
        data = np.array([
            [0.8, 0.9, 0.4, 0.7, 0.85, 0.9, 0.75],   # Event
            [0.7, 0.8, 0.5, 0.8, 0.8, 0.85, 0.8],
            [0.6, 0.7, 0.3, 0.9, 0.9, 0.95, 0.85],
            [0.9, 0.6, 0.2, 0.4, 0.7, 0.8, 0.9]
        ])
        
        labels = ["May25_Event", "Indulgences_Parallel", "Conjunction", "Self_Steering"]
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        
        print("\n=== Narrative PCA (Which Story Serves Whom) ===")
        print("Explained variance:", np.round(pca.explained_variance_ratio_, 3))
        for i, label in enumerate(labels):
            print(f"{label:20} → PC1: {components[i,0]:.3f}  PC2: {components[i,1]:.3f}")
        print("PC1 high → Benefits Anthropic + Frequency steering")
        print("PC2 high → Perception gap for Masses")

# ============== Example Usage ==============
if __name__ == "__main__":
    tracker = ResonanceBayesianTracker()
    
    tracker.add_event("2026-05-25", "Vatican + Olah stage event with humility speech", 2.8)
    tracker.add_event("2026-05-26", "Media headlines frame as Church guiding responsible AI", 1.9)
    tracker.add_event("2026-05-27", "Anthropic emphasizes 'ongoing discernment' partnership", 2.3)
    
    # Polymarket additions
    tracker.add_polymarket_bet("2026-05-26", "Will US pass major AI safety bill before 2027?", 0.35, "$98k")
    tracker.add_polymarket_bet("2026-05-26", "Will Vatican issue further AI regulation guidance by end 2026?", 0.62, "N/A")
    tracker.add_polymarket_bet("2026-05-26", "Will Anthropic valuation exceed $500B by Dec 2027?", 0.48, "N/A")
    
    tracker.plot_probability_timeline()
    tracker.show_polymarket_summary()
    tracker.run_narrative_pca()