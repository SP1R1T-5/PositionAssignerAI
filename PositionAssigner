import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class MissionMatchConfig:
    """Configuration for Mission Match AI."""
    num_employees: int = 500
    random_seed: int = 42

    # Organization structure
    departments: List[str] = field(
        default_factory=lambda: [
            "Operations",
            "Technology",
            "Sales",
            "Marketing",
            "Finance",
            "HR",
        ]
    )
    job_levels: List[str] = field(
        default_factory=lambda: [
            "Entry",
            "Associate",
            "Senior",
            "Lead",
            "Manager",
        ]
    )

    # Employee characteristic ranges
    age_range: Tuple[int, int] = (22, 65)
    experience_max: int = 35
    performance_range: Tuple[int, int] = (40, 100)
    performance_average: float = 75.0
    performance_std: float = 15.0

    # Skill and training hour ranges
    technical_skill_range: Tuple[int, int] = (3, 10)
    leadership_skill_range: Tuple[int, int] = (2, 9)
    communication_skill_range: Tuple[int, int] = (4, 10)
    training_hours_range: Tuple[int, int] = (10, 120)

    # Retention calculation
    base_retention_rate: float = 0.7
    performance_retention_factor: float = 0.01
    experience_retention_factor: float = 0.02
    max_experience_bonus: float = 0.2

    # Filtering criteria
    leadership_criteria: Dict[str, int] = field(
        default_factory=lambda: {"min_leadership_skill": 7, "min_performance": 75}
    )
    technical_criteria: Dict[str, int] = field(
        default_factory=lambda: {"min_technical_skill": 8, "min_performance": 70}
    )
    senior_criteria: Dict[str, int] = field(
        default_factory=lambda: {"min_experience": 8, "min_performance": 80}
    )
    top_performer_threshold: int = 85

    # Clustering & risk
    num_clusters: int = 4
    risk_thresholds: List[float] = field(default_factory=lambda: [0.4, 0.7, 1.0])
    risk_labels: List[str] = field(
        default_factory=lambda: ["High Risk", "Medium Risk", "Low Risk"]
    )

    # Feature selection
    retention_features: List[str] = field(
        default_factory=lambda: [
            "Age", "Experience", "Performance", "Technical_Skill",
            "Leadership_Skill", "Communication_Skill", "Training_Hours",
            "Job_Level_Encoded", "Department_Encoded",
        ]
    )
    clustering_features: List[str] = field(
        default_factory=lambda: [
            "Age", "Experience", "Performance",
            "Technical_Skill", "Leadership_Skill", "Communication_Skill",
        ]
    )


class MissionMatchAI:
    """Core Mission Match AI functionality."""
    def __init__(self, config: Optional[MissionMatchConfig] = None) -> None:
        self.config = config or MissionMatchConfig()
        self.df: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.retention_model: Optional[LogisticRegression] = None
        self.clusterer: Optional[KMeans] = None

    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV or fallback to synthetic data."""
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} records from '{path}'")
            self.df = df
        except Exception as e:
            print(f"Error loading '{path}': {e}\nGenerating synthetic data.")
            self.df = self.generate_data()
        return self.df

    def generate_data(self, n: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic employee records."""
        n = n or self.config.num_employees
        np.random.seed(self.config.random_seed)

        rows = []
        for i in range(n):
            age = np.random.randint(*self.config.age_range)
            exp = min(np.random.randint(0, age - 21), self.config.experience_max)
            perf = np.clip(
                np.random.normal(self.config.performance_average, self.config.performance_std),
                *self.config.performance_range
            )
            tech = np.random.randint(*self.config.technical_skill_range)
            lead = np.random.randint(*self.config.leadership_skill_range)
            comm = np.random.randint(*self.config.communication_skill_range)
            train = np.random.randint(*self.config.training_hours_range)

            retention_prob = (
                self.config.base_retention_rate
                + (perf - 70) * self.config.performance_retention_factor
                + min(exp * self.config.experience_retention_factor,
                      self.config.max_experience_bonus)
            )
            retain = int(np.random.random() < retention_prob)

            rows.append({
                "Employee_ID": f"EMP_{i:04d}",
                "Age": age,
                "Experience": exp,
                "Job_Level": np.random.choice(self.config.job_levels),
                "Department": np.random.choice(self.config.departments),
                "Performance": round(perf, 1),
                "Technical_Skill": tech,
                "Leadership_Skill": lead,
                "Communication_Skill": comm,
                "Training_Hours": train,
                "Retention": retain,
            })

        df = pd.DataFrame(rows)
        print(f"Generated {n} synthetic records")
        self.df = df
        return df

    def preprocess(self) -> None:
        """Label-encode categorical columns."""
        if self.df is None:
            raise ValueError("No data to preprocess")

        for col in ("Job_Level", "Department"):
            le = self.label_encoders.get(col, LabelEncoder())
            self.df[f"{col}_Encoded"] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        print("Preprocessing complete.")

    def train_retention(self) -> LogisticRegression:
        """Train a logistic regression retention model."""
        if self.df is None:
            self.generate_data()

        self.preprocess()
        feats = [c for c in self.config.retention_features if c in self.df]
        X = self.df[feats]
        y = self.df.get("Retention", pd.Series(np.ones(len(X)), dtype=int))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_seed
        )

        model = LogisticRegression(max_iter=1_000, random_state=self.config.random_seed)
        model.fit(X_train, y_train)
        self.retention_model = model

        if y.nunique() > 1:
            y_pred = model.predict(X_test)
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(classification_report(y_test, y_pred))

        return model

    def cluster(self, k: Optional[int] = None) -> pd.DataFrame:
        """Cluster employees into k groups."""
        if self.df is None:
            raise ValueError("No data to cluster")

        k = k or self.config.num_clusters
        feats = [c for c in self.config.clustering_features if c in self.df]
        X = self.scaler.fit_transform(self.df[feats])

        km = KMeans(n_clusters=k, random_state=self.config.random_seed)
        self.df["Cluster"] = km.fit_predict(X)
        self.clusterer = km

        summary = (
            self.df
            .groupby("Cluster")[feats + (["Retention"] if "Retention" in self.df else [])]
            .mean()
            .round(2)
        )
        print(f"Cluster summary (k={k}):\n{summary}")
        return self.df

    def find_top(self, role: str = "leadership") -> pd.DataFrame:
        """Return the top 10 candidates for a given role."""
        if self.df is None:
            raise ValueError("No data available")

        criteria = {
            "leadership": self.config.leadership_criteria,
            "technical": self.config.technical_criteria,
            "senior": self.config.senior_criteria,
        }.get(role, {})

        df = self.df
        for key, val in criteria.items():
            df = df[df[key.replace("min_", "").title()] >= val]

        return df.nlargest(10, "Performance")[[
            "Employee_ID", "Job_Level", "Department", "Performance",
            "Technical_Skill", "Leadership_Skill"
        ]]

    def predict_risk(self) -> pd.DataFrame:
        """Predict retention risk for each employee."""
        if not self.retention_model:
            raise RuntimeError("Train the retention model first")

        feats = [c for c in self.config.retention_features if c in self.df]
        probs = self.retention_model.predict_proba(self.df[feats])[:, 1]

        result = self.df[[
            col for col in ["Employee_ID", "Job_Level", "Department", "Performance"]
            if col in self.df
        ]].copy()
        result["Retention_Probability"] = probs
        result["Risk_Level"] = pd.cut(
            probs,
            bins=[0] + self.config.risk_thresholds,
            labels=self.config.risk_labels
        )
        return result.sort_values("Retention_Probability")

    def visualize(self) -> None:
        """Plot retention and performance insights."""
        if self.df is None:
            raise ValueError("No data to visualize")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Retention by Department
        if {"Department", "Retention"}.issubset(self.df):
            self.df.groupby("Department")["Retention"].mean().plot(
                kind="bar", ax=axes[0, 0]
            )
            axes[0, 0].set_title("Retention Rate by Department")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Performance distribution
        axs = axes[0, 1]
        if "Performance" in self.df:
            self.df["Performance"].hist(ax=axs, bins=20, alpha=0.7)
            axs.set_title("Performance Distribution")

        # Experience vs Performance
        ax = axes[1, 0]
        ax.scatter(
            self.df["Experience"], self.df["Performance"],
            c=self.df.get("Retention", self.df["Age"]), alpha=0.6
        )
        ax.set_xlabel("Experience (years)")
        ax.set_ylabel("Performance")
        ax.set_title("Experience vs. Performance")

        # Skills by Job Level
        if "Job_Level" in self.df:
            skills = ["Technical_Skill", "Leadership_Skill"]
            valid = [s for s in skills if s in self.df]
            self.df.groupby("Job_Level")[valid].mean().plot(
                kind="bar", ax=axes[1, 1]
            )
            axes[1, 1].set_title("Average Skills by Job Level")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def report(self) -> str:
        """Generate a text summary report."""
        if self.df is None:
            return "No data available."

        lines = [
            "=== MISSION MATCH AI REPORT ===",
            f"Total Employees: {len(self.df)}",
        ]

        if "Retention" in self.df:
            lines.append(f"Overall Retention: {self.df['Retention'].mean():.2%}")
        if "Performance" in self.df:
            avg_perf = self.df["Performance"].mean()
            lines.append(f"Avg. Performance: {avg_perf:.1f}")
            top = self.df.nlargest(5, "Performance")
            lines.append("Top 5 Performers:")
            lines += [f"  {row.Employee_ID}: {row.Performance:.1f}"
                      for _, row in top.iterrows()]

        if {"Department", "Retention"}.issubset(self.df):
            best = self.df.groupby("Department")["Retention"].mean().idxmax()
            rate = self.df.groupby("Department")["Retention"].mean().max()
            lines.append(f"Best Dept. by Retention: {best} ({rate:.1%})")

        return "\n".join(lines)


def main():
    ai = MissionMatchAI()
    ai.generate_data()
    ai.train_retention()
    ai.cluster()

    # Top-5 for each chart
    risk_df = ai.predict_risk()
    top_retention = risk_df.nlargest(5, "Retention_Probability")
    print("\nTop 5 by Retention Probability:")
    print(top_retention[["Employee_ID", "Retention_Probability", "Risk_Level"]])

    top_performers = ai.df.nlargest(5, "Performance")[["Employee_ID", "Performance"]]
    print("\nTop 5 Performers:")
    print(top_performers)

    top_experience = ai.df.nlargest(5, "Experience")[["Employee_ID", "Experience"]]
    print("\nTop 5 by Experience:")
    print(top_experience)

    ai.df["Total_Skill"] = ai.df["Technical_Skill"] + ai.df["Leadership_Skill"]
    top_skilled = ai.df.nlargest(5, "Total_Skill")[
        ["Employee_ID", "Technical_Skill", "Leadership_Skill", "Total_Skill"]
    ]
    print("\nTop 5 by Combined Skill:")
    print(top_skilled)

    print("\n=== GENERATING VISUALIZATIONS ===")
    ai.visualize()

    print("\n" + ai.report())


if __name__ == "__main__":
    main()
