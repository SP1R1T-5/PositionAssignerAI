import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Mapping for risk tolerance strings to numeric values
RISK_MAP = {
    'low': 0.25,
    'medium': 0.50,
    'high': 0.75
}

@dataclass
class CandidateProfile:
    id: str                                      # Setting ID Number for employment candidates
    age: int                                     # Candidate's age 
    experience: int                              # Candidate's work experience 
    job_titles: List[Tuple[str, int]]            # Candidate's work title, years
    core_skills: List[str]                       # Candidate's core skills
    certifications: List[str]                    # Candidate's certifications
    education: Dict[str, Any]                    # Candidate's education history {'degree': str, 'field': str, 'institution': str}
    languages: List[str]                         # Candidate's known spoken and programming langaugaes known
    projects: List[Dict[str, Any]]               # Candidate's previous projects {'name': str, 'role': str, 'outcome': str}
    leadership_years: int                        # Candidate's experience in positions of leadership
    big_five: Dict[str, float]                   # Candidate's e.g., {'Openness': 0.8, ...}
    cognitive_style: str                         # Candidate's ideal working style {Logical, Creative, Analytical}
    conflict_style: str                          # Candidate's conflict resolution style {Avoidant, Competitive, Collaborative}
    work_pref: str                               # Candidate's work environment preference {Balanced, Flexible, Structured}
    team_roles: List[str]                        # Candidate's team role tendency {Resource Investigator,Implementer, Completer-Finisher}
    risk_tolerance: float                        # Candidate's risk tolerance {Low, Medium, High}
    industry_pref: List[str]                     # Candidate's preferred industry {Education, Retail, Technology}
    location_pref: str                           # Candidate's preferred working location {On-Site, Hybrid, Remote} 
    team_size_pref: Tuple[int, int]              # Candidate's ideal team size {Small, Medium, Large}
    career_goal: str                             # Candidate's career aspiration {Technical, Management, Innovation}
    performance: float = 0.0                     # Candidate's potential performance based on resume and interviews
    retention_prob: float = 0.0                  # Candidate's retention probability
    cluster: int = -1                            # Candidate's cluster group

@dataclass
class Config:
    random_seed: int = 42                        # Random seed for reproducibility
    n_clusters: int = 3                          # Number of clusters for KMeans

class MissionMatchAI:
    def __init__(self, cfg: Config):              # Initialize AI system: seed RNG, prepare storage, scaler, classifier, and clusterer
        np.random.seed(cfg.random_seed)
        self.profiles: List[CandidateProfile] = []
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500)
        self.clusterer = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_seed)

    def load_profiles(self, csv_path: str):       # Load and parse CSV into CandidateProfile instances
        df = pd.read_csv(csv_path).replace({np.nan: None})
        print(f"Found columns in CSV: {list(df.columns)}")

        column_mapping = { # Setting the columns from dataset to variables
            'Name': 'id',
            'Job History': 'job_titles',
            'Core Skills': 'core_skills',
            'Certifications': 'certifications',
            'Education': 'education',
            'Languages': 'languages',
            'Project History': 'projects',
            'Leadership Experience': 'leadership_years',
            'Cognitive Style': 'cognitive_style',
            'Conflict Resolution Style': 'conflict_style',
            'Work Environment Preference': 'work_pref',
            'Team Role Tendency': 'team_roles',
            'Risk Tolerance': 'risk_tolerance',
            'Preferred Industry': 'industry_pref',
            'Geographic Preference': 'location_pref',
            'Ideal Team Size': 'team_size_pref',
            'Career Aspiration': 'career_goal',
            'Matched Role': 'performance'
        }
        df = df.rename(columns=column_mapping)
        profs = []

        for idx, rec in df.iterrows():
            profile_data = {              # Prefered candidate profile for the position 
                'id': rec.get('id', f'candidate_{idx}'),
                'age': 30,
                'experience': 5,
                'job_titles': self._parse_list_field(rec.get('job_titles', '')),
                'core_skills': self._parse_list_field(rec.get('core_skills', '')),
                'certifications': self._parse_list_field(rec.get('certifications', '')),
                'languages': self._parse_list_field(rec.get('languages', '')),
                'projects': self._parse_list_field(rec.get('projects', '')),
                'team_roles': self._parse_list_field(rec.get('team_roles', '')),
                'industry_pref': self._parse_list_field(rec.get('industry_pref', '')),
                'education': self._parse_dict_field(rec.get('education', '')),
                'big_five': {
                    'openness': float(rec.get('Openness', 0.5)),
                    'conscientiousness': float(rec.get('Conscientiousness', 0.5)),
                    'extraversion': float(rec.get('Extraversion', 0.5)),
                    'agreeableness': float(rec.get('Agreeableness', 0.5)),
                    'neuroticism': float(rec.get('Neuroticism', 0.5))
                },
                'leadership_years': self._parse_numeric_field(rec.get('leadership_years', '0')),
            }
            raw_risk = rec.get('risk_tolerance', None)
            if isinstance(raw_risk, str):
                profile_data['risk_tolerance'] = RISK_MAP.get(raw_risk.strip().lower(), 0.5)    # Sets string values to intager, if there isn't then it defaults to .5
            else:
                try:
                    profile_data['risk_tolerance'] = float(raw_risk)    # If numeric then convert it directly
                except:
                    profile_data['risk_tolerance'] = 0.5                # If non-numeric then convert to .5

            profile_data.update({
                'cognitive_style': rec.get('cognitive_style', 'Unknown'), # Candidate’s preferred thinking/process style
                'conflict_style': rec.get('conflict_style', 'Unknown'), # Candidate’s conflict resolution approach
                'work_pref': rec.get('work_pref', 'Unknown'), # Candidate’s ideal work environment (Structured, Flexible, etc.)
                'location_pref': rec.get('location_pref', 'Unknown'),  # Candidate’s geographic/remote work preference
                'career_goal': rec.get('career_goal', 'Unknown'), # Candidate’s stated career aspiration
                'team_size_pref': self._parse_team_size(rec.get('team_size_pref', '5-10')), # Parse ideal team size range (e.g., “5-10”)
                'performance': self._calculate_performance(rec), # Compute performance score using skills, certifications, etc.
                'retention_prob': 0.0,  # Initialize retention probability to zero (to be trained later)
                'cluster': -1  # Initialize cluster assignment to -1 (unclustered)
            })

            try:
                profs.append(CandidateProfile(**profile_data))
            except Exception as e:
                print(f"Warning: Could not create profile for record {idx}: {e}")

        self.profiles = profs
        print(f"Successfully loaded {len(self.profiles)} profiles")

    def _parse_list_field(self, field_str):        # Convert CSV field to Python list, handling literal and comma-sep formats
        if not field_str or pd.isna(field_str):
            return []
        if isinstance(field_str, str) and field_str.startswith('[') and field_str.endswith(']'):
            try:
                return ast.literal_eval(field_str)
            except:
                pass
        return [item.strip() for item in str(field_str).split(',') if item.strip()]

    def _parse_dict_field(self, field_str):        # Convert CSV field to dict, handling JSON-style strings
        if not field_str or pd.isna(field_str):
            return {}
        if isinstance(field_str, str) and field_str.startswith('{') and field_str.endswith('}'):
            try:
                return ast.literal_eval(field_str)
            except:
                pass
        return {'degree': str(field_str)}

    def _parse_numeric_field(self, field_str):     # Safely parse a string/number field into an integer
        try:
            return int(float(str(field_str)))
        except:
            return 0

    def _parse_team_size(self, team_size_str):     # Parse ranges like "5-10" or "5 to 10" into a (min, max) tuple
        try:
            s = str(team_size_str)
            if '-' in s:
                a, b = s.split('-')
                return int(a), int(b)
            if 'to' in s.lower():
                a, b = s.lower().split('to')
                return int(a), int(b)
        except:
            pass
        return 1, 10

    def _calculate_performance(self, rec):        # Heuristic scoring based on skills, certs, leadership, and languages
        score = 50.0
        score += len(self._parse_list_field(rec.get('core_skills', ''))) * 2
        score += len(self._parse_list_field(rec.get('certifications', ''))) * 5
        score += self._parse_numeric_field(rec.get('leadership_years', '0')) * 3
        score += len(self._parse_list_field(rec.get('languages', ''))) * 2
        return min(score, 100.0)

    def train_retention(self):                     # Train logistic regression to predict retention from performance & leadership
        if not self.profiles:
            print('No profiles available for training')
            return
        df = pd.DataFrame([vars(p) for p in self.profiles])
        if df.empty:
            print('Skipping retention (no data)')
            return
        X = df[['performance', 'leadership_years']]
        y = (df['performance'] > 80).astype(int)
        if y.nunique() < 2:
            print(f"Skipping retention training: only one class present ({y.unique()[0]})")
            for p in self.profiles:
                p.retention_prob = float(y.unique()[0])
            return
        self.model.fit(X, y)
        probs = self.model.predict_proba(X)[:, 1]
        for p, prob in zip(self.profiles, probs):
            p.retention_prob = prob
        print(f"Trained retention model on {len(self.profiles)} profiles")
        print(f"Average retention probability: {np.mean(probs):.3f}")

    def cluster(self):                             # Scale features and K-means cluster candidates into cfg.n_clusters groups
        if not self.profiles:
            print('Skipping cluster (no profiles)')
            return
        df = pd.DataFrame({
            'skill_count': [len(p.core_skills) for p in self.profiles],
            'risk_tolerance': [p.risk_tolerance for p in self.profiles],
            'performance': [p.performance for p in self.profiles],
            'leadership_years': [p.leadership_years for p in self.profiles]
        })
        Xs = self.scaler.fit_transform(df)
        labels = self.clusterer.fit_predict(Xs)
        for p, lbl in zip(self.profiles, labels):
            p.cluster = int(lbl)
        print(f"Clustered {len(self.profiles)} profiles into {len(set(labels))} clusters")
        for cid, count in pd.Series(labels).value_counts().sort_index().items():
            print(f"  Cluster {cid}: {count} candidates")

    def analyze_clusters(self):                    # Print summary statistics for each candidate cluster
        if not self.profiles or all(p.cluster == -1 for p in self.profiles):
            print('No clustering data available')
            return
        print("\nCLUSTER ANALYSIS")
        for cid in sorted(set(p.cluster for p in self.profiles)):
            grp = [p for p in self.profiles if p.cluster == cid]
            print(f"\nCluster {cid} ({len(grp)} candidates):")
            print(f"  Avg Performance: {np.mean([p.performance for p in grp]):.1f}")
            print(f"  Avg Leadership: {np.mean([p.leadership_years for p in grp]):.1f}")
            print(f"  Avg Risk Tolerance: {np.mean([p.risk_tolerance for p in grp]):.2f}")
            print("  Members:", ", ".join(p.id for p in grp))

    def generate_report(self):                     # Output overall counts, performance range, and call cluster analysis
        if not self.profiles:
            print('No profiles available')
            return
        print("\nREPORT")
        print(f"Total Candidates: {len(self.profiles)}")
        perf = [p.performance for p in self.profiles]
        print(f"Performance Range: {min(perf):.1f} - {max(perf):.1f}")
        self.analyze_clusters()

    def plot_analytics(self):                      # Display matplotlib charts: cluster sizes, distributions, and scatter/boxplots
        # Number of candidates per cluster
        cluster_counts = pd.Series([p.cluster for p in self.profiles]).value_counts().sort_index()
        plt.figure()
        plt.bar([f'Cluster {cid}' for cid in cluster_counts.index], cluster_counts.values)
        plt.title('Number of Candidates per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Candidates')
        plt.show()

        # Performance distribution
        perf = [p.performance for p in self.profiles]
        plt.figure()
        plt.hist(perf, bins=10)
        plt.title('Performance Distribution')
        plt.xlabel('Performance Score')
        plt.ylabel('Number of Candidates')
        plt.show()

        # Retention probability distribution
        probs = [p.retention_prob for p in self.profiles]
        plt.figure()
        plt.hist(probs, bins=10)
        plt.title('Retention Probability Distribution')
        plt.xlabel('Retention Probability')
        plt.ylabel('Number of Candidates')
        plt.show()

        # Cluster scatter (Risk vs Performance)
        df_plot = pd.DataFrame({
            'risk_tolerance': [p.risk_tolerance for p in self.profiles],
            'performance':     [p.performance    for p in self.profiles],
            'cluster':         [p.cluster        for p in self.profiles]
        })
        plt.figure()
        for cid in sorted(df_plot['cluster'].unique()):
            subset = df_plot[df_plot['cluster'] == cid]
            plt.scatter(subset['risk_tolerance'], subset['performance'], label=f'Cluster {cid}')
        plt.title('Clusters: Risk vs Performance')
        plt.xlabel('Risk Tolerance')
        plt.ylabel('Performance')
        plt.legend()
        plt.show()

        # Boxplot of Performance by Cluster
        plt.figure()
        data_perf = [
            df_plot[df_plot['cluster'] == cid]['performance'].values
            for cid in sorted(df_plot['cluster'].unique())
        ]
        plt.boxplot(data_perf, labels=[f'Cluster {cid}' for cid in sorted(df_plot['cluster'].unique())])
        plt.title('Performance by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Performance Score')
        plt.show()

        # Boxplot of Risk Tolerance by Cluster
        plt.figure()
        data_risk = [
            df_plot[df_plot['cluster'] == cid]['risk_tolerance'].values
            for cid in sorted(df_plot['cluster'].unique())
        ]
        plt.boxplot(data_risk, labels=[f'Cluster {cid}' for cid in sorted(df_plot['cluster'].unique())])
        plt.title('Risk Tolerance by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Risk Tolerance')
        plt.show()


if __name__ == "__main__":
    print("Initializing MissionMatchAI...")       # Entry point: initialize and run analysis pipeline
    config = Config()
    ai_system = MissionMatchAI(config)

    print("Loading profiles...")                  # Step 1: load candidate data
    ai_system.load_profiles("MissionMatchDS1.csv")

    print("\nTraining retention model...")        # Step 2: train retention predictor
    ai_system.train_retention()

    print("\nClustering candidates...")           # Step 3: cluster profiles
    ai_system.cluster()

    print("\nGenerating report...")               # Step 4: summarize clusters and performance
    ai_system.generate_report()

    print("\nDisplaying graphical analytics...")  # Step 5: visualize results
    ai_system.plot_analytics()

    print("\nAnalysis complete!")                 # Finish execution

#Real Jon Fortnite
