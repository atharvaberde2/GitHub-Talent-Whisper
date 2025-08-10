#!/usr/bin/env python3
"""
🏆 TOURNAMENT-GRADE ML Pattern Recognition Engine (Lite Version)
Advanced psychological profiling and behavioral analysis for developer talent detection
This is a lightweight version that doesn't require spaCy
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import statistics
import textstat

class AdvancedPatternEngine:
    """Tournament-grade ML pattern recognition for developer psychology"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Psychological trait models
        self.personality_keywords = {
            'openness': ['creative', 'innovative', 'experimental', 'novel', 'artistic', 'imaginative'],
            'conscientiousness': ['organized', 'systematic', 'thorough', 'careful', 'detailed', 'structured'],
            'extraversion': ['collaborative', 'team', 'community', 'social', 'outgoing', 'active'],
            'agreeableness': ['helpful', 'supportive', 'cooperative', 'friendly', 'kind', 'considerate'],
            'neuroticism': ['stress', 'urgent', 'critical', 'issue', 'problem', 'error']
        }
        
        # Technical sophistication patterns
        self.technical_patterns = {
            'architecture': ['design', 'pattern', 'architecture', 'structure', 'framework', 'system'],
            'optimization': ['performance', 'optimize', 'efficient', 'fast', 'memory', 'algorithm'],
            'security': ['secure', 'auth', 'encrypt', 'validate', 'sanitize', 'vulnerability'],
            'scalability': ['scale', 'distributed', 'concurrent', 'parallel', 'load', 'capacity']
        }
        
    def extract_ml_features(self, commits: List[Dict], prs: List[Dict], repos: List[Dict]) -> Dict[str, Any]:
        """Extract sophisticated ML features from all data sources"""
        
        features = {}
        
        # 1. Commit Pattern Analysis with ML
        features.update(self._analyze_commit_patterns_ml(commits))
        
        # 2. Communication Sophistication Analysis
        pr_texts = [f"{pr.get('title', '')} {pr.get('body', '')}" for pr in prs if pr.get('title') or pr.get('body')]
        features.update(self._analyze_communication_sophistication(pr_texts))
        
        # 3. Technical Depth Analysis
        features.update(self._analyze_technical_depth(commits, prs, repos))
        
        # 4. Behavioral Pattern Recognition
        features.update(self._analyze_behavioral_patterns(commits, prs))
        
        # 5. Innovation & Learning Patterns
        features.update(self._analyze_innovation_patterns(repos, commits))
        
        return features
    
    def _analyze_commit_patterns_ml(self, commits: List[Dict]) -> Dict[str, Any]:
        """ML-powered commit pattern analysis"""
        if not commits:
            return {"commit_ml_features": {}}
            
        commit_messages = [commit.get('message', '') for commit in commits]
        commit_times = []
        commit_sizes = []
        
        for commit in commits:
            # Parse commit time
            try:
                commit_time = datetime.fromisoformat(commit.get('date', '').replace('Z', '+00:00'))
                commit_times.append(commit_time)
            except:
                pass
                
            # Estimate commit size (rough proxy)
            message_length = len(commit.get('message', ''))
            commit_sizes.append(message_length)
        
        # Vectorize commit messages for clustering
        commit_diversity = 0
        if commit_messages:
            try:
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(commit_messages)
                
                # Cluster commits to find patterns
                n_clusters = min(5, len(commit_messages))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                    cluster_distribution = np.bincount(clusters) / len(clusters)
                    commit_diversity = stats.entropy(cluster_distribution)
            except:
                commit_diversity = 0
        
        # Temporal patterns
        temporal_features = {}
        if commit_times:
            hours = [ct.hour for ct in commit_times]
            days = [ct.weekday() for ct in commit_times]
            
            # Work pattern consistency
            work_hour_ratio = len([h for h in hours if 9 <= h <= 17]) / len(hours)
            weekend_ratio = len([d for d in days if d >= 5]) / len(days)
            
            # Commit frequency patterns
            if len(commit_times) > 1:
                time_diffs = [(commit_times[i] - commit_times[i-1]).total_seconds() / 3600 
                             for i in range(1, len(commit_times))]
                commit_frequency_std = np.std(time_diffs) if time_diffs else 0
            else:
                commit_frequency_std = 0
                
            temporal_features = {
                'work_hour_ratio': work_hour_ratio,
                'weekend_ratio': weekend_ratio,
                'commit_frequency_consistency': 1 / (1 + commit_frequency_std)  # Higher = more consistent
            }
        
        return {
            "commit_ml_features": {
                "message_diversity": commit_diversity,
                "avg_commit_size": statistics.mean(commit_sizes) if commit_sizes else 0,
                "commit_size_variance": statistics.variance(commit_sizes) if len(commit_sizes) > 1 else 0,
                **temporal_features
            }
        }
    
    def _analyze_communication_sophistication(self, texts: List[str]) -> Dict[str, Any]:
        """Advanced NLP analysis of communication patterns"""
        if not texts:
            return {"communication_sophistication": {}}
        
        # Clean and prepare texts
        clean_texts = [self._clean_text(text) for text in texts if text.strip()]
        if not clean_texts:
            return {"communication_sophistication": {}}
        
        # 1. Readability and complexity metrics
        readability_scores = []
        complexity_scores = []
        
        for text in clean_texts:
            if len(text) > 10:  # Only analyze substantial texts
                # Readability
                try:
                    flesch_score = textstat.flesch_reading_ease(text)
                    readability_scores.append(flesch_score)
                except:
                    readability_scores.append(50)  # Default neutral score
                
                # Complexity (vocabulary richness)
                words = text.lower().split()
                unique_words = set(words)
                complexity = len(unique_words) / len(words) if words else 0
                complexity_scores.append(complexity)
        
        # 2. Psychological trait analysis
        psychological_traits = self._analyze_psychological_traits(clean_texts)
        
        # 3. Technical communication analysis
        technical_sophistication = self._analyze_technical_communication(clean_texts)
        
        # Aggregate scores
        avg_readability = statistics.mean(readability_scores) if readability_scores else 50
        avg_complexity = statistics.mean(complexity_scores) if complexity_scores else 0
        
        # Sentiment consistency (simplified version)
        sentiment_consistency = 0.8  # Default good consistency
        
        return {
            "communication_sophistication": {
                "readability_score": avg_readability,
                "vocabulary_richness": avg_complexity,
                "sentiment_consistency": sentiment_consistency,
                "psychological_traits": psychological_traits,
                "technical_sophistication": technical_sophistication,
                "communication_volume": len(clean_texts)
            }
        }
    
    def _analyze_psychological_traits(self, texts: List[str]) -> Dict[str, float]:
        """Analyze Big Five personality traits from text"""
        combined_text = ' '.join(texts).lower()
        words = combined_text.split()
        word_count = len(words)
        
        trait_scores = {}
        for trait, keywords in self.personality_keywords.items():
            matches = sum(1 for word in words if word in keywords)
            # Normalize by text length and scale to 0-100
            trait_scores[trait] = min(100, (matches / word_count * 1000)) if word_count > 0 else 0
        
        return trait_scores
    
    def _analyze_technical_communication(self, texts: List[str]) -> Dict[str, float]:
        """Analyze technical communication sophistication"""
        combined_text = ' '.join(texts).lower()
        words = combined_text.split()
        word_count = len(words)
        
        technical_scores = {}
        for category, keywords in self.technical_patterns.items():
            matches = sum(1 for word in words if word in keywords)
            technical_scores[f"{category}_sophistication"] = min(100, (matches / word_count * 1000)) if word_count > 0 else 0
        
        # Overall technical communication score
        technical_scores['overall_technical_score'] = statistics.mean(technical_scores.values()) if technical_scores else 0
        
        return technical_scores
    
    def _analyze_technical_depth(self, commits: List[Dict], prs: List[Dict], repos: List[Dict]) -> Dict[str, Any]:
        """Analyze technical depth and expertise"""
        
        # Language diversity and expertise
        languages = {}
        for repo in repos:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        language_diversity = len(languages)
        primary_language_dominance = max(languages.values()) / sum(languages.values()) if languages else 0
        
        # Technology adoption patterns
        tech_keywords = [
            'typescript', 'rust', 'go', 'kotlin', 'swift', 'webassembly',
            'docker', 'kubernetes', 'microservices', 'graphql', 'grpc',
            'redis', 'elasticsearch', 'mongodb', 'postgresql',
            'react', 'vue', 'angular', 'svelte', 'next.js', 'nuxt',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas'
        ]
        
        tech_adoption_score = 0
        for repo in repos:
            repo_text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
            for tech in tech_keywords:
                if tech in repo_text:
                    tech_adoption_score += 1
        
        tech_adoption_score = min(100, tech_adoption_score * 5)  # Scale to 0-100
        
        return {
            "technical_depth": {
                "language_diversity": language_diversity,
                "specialization_focus": primary_language_dominance,
                "technology_adoption": tech_adoption_score,
                "repository_count": len(repos)
            }
        }
    
    def _analyze_behavioral_patterns(self, commits: List[Dict], prs: List[Dict]) -> Dict[str, Any]:
        """Analyze behavioral patterns and work style"""
        
        # Collaboration patterns
        pr_descriptions = [pr.get('body', '') for pr in prs if pr.get('body')]
        avg_pr_description_length = statistics.mean([len(desc) for desc in pr_descriptions]) if pr_descriptions else 0
        
        # Detailed communication tendency
        detailed_communication_score = min(100, avg_pr_description_length / 10)  # Scale to 0-100
        
        # Consistency patterns
        commit_messages = [commit.get('message', '') for commit in commits]
        commit_message_lengths = [len(msg) for msg in commit_messages]
        
        consistency_score = 0
        if len(commit_message_lengths) > 1:
            mean_length = statistics.mean(commit_message_lengths)
            if mean_length > 0:
                cv = statistics.stdev(commit_message_lengths) / mean_length
                consistency_score = max(0, 100 - cv * 20)  # Higher consistency = lower coefficient of variation
            else:
                consistency_score = 50  # Default moderate consistency
        
        # Simple collaboration tendency based on PR activity
        collaboration_tendency = min(100, len(prs) * 10)  # Scale PR count to 0-100
        
        return {
            "behavioral_patterns": {
                "detailed_communication": detailed_communication_score,
                "consistency_score": consistency_score,
                "collaboration_tendency": collaboration_tendency
            }
        }
    
    def _analyze_innovation_patterns(self, repos: List[Dict], commits: List[Dict]) -> Dict[str, Any]:
        """Analyze innovation and learning patterns"""
        
        # Recent activity innovation
        current_year = datetime.now().year
        recent_repos = 0
        
        for repo in repos:
            try:
                created_at = datetime.fromisoformat(repo.get('created_at', '').replace('Z', '+00:00'))
                if created_at.year >= current_year - 1:
                    recent_repos += 1
            except:
                pass
        
        innovation_velocity = (recent_repos / len(repos) * 100) if repos else 0
        
        # Learning pattern analysis from commit messages
        learning_keywords = ['learn', 'study', 'experiment', 'try', 'explore', 'research', 'investigate']
        commit_messages = [commit.get('message', '').lower() for commit in commits]
        
        learning_signals = 0
        for message in commit_messages:
            for keyword in learning_keywords:
                if keyword in message:
                    learning_signals += 1
                    break
        
        learning_orientation = (learning_signals / len(commits) * 100) if commits else 0
        
        return {
            "innovation_patterns": {
                "innovation_velocity": innovation_velocity,
                "learning_orientation": learning_orientation,
                "experimentation_score": min(100, (recent_repos + learning_signals) * 5)
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove URLs, code blocks, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_confidence_with_statistical_validation(self, scores: List[float], sample_size: int) -> Dict[str, float]:
        """Calculate confidence with statistical validation"""
        if not scores or len(scores) < 2:
            return {"confidence": 0, "significance": "low", "sample_adequacy": "insufficient"}
        
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)
        
        # Calculate confidence interval
        confidence_interval = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=std_error)
        
        # Determine statistical significance
        if len(scores) >= 30:
            significance = "high"
        elif len(scores) >= 10:
            significance = "medium"
        else:
            significance = "low"
        
        # Sample adequacy
        if sample_size >= 50:
            adequacy = "excellent"
        elif sample_size >= 20:
            adequacy = "good"
        elif sample_size >= 10:
            adequacy = "fair"
        else:
            adequacy = "limited"
        
        # Calculate final confidence score
        base_confidence = min(95, 60 + (len(scores) * 2))
        
        # Adjust for statistical factors
        interval_width = confidence_interval[1] - confidence_interval[0]
        precision_bonus = max(0, 20 - interval_width)
        
        final_confidence = min(98, base_confidence + precision_bonus)
        
        return {
            "confidence": final_confidence,
            "significance": significance,
            "sample_adequacy": adequacy,
            "confidence_interval": confidence_interval,
            "precision_score": precision_bonus
        }
    
    def generate_ml_insights(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Generate ML-powered insights from features"""
        
        insights = {}
        
        # Communication sophistication insight
        comm_features = features.get('communication_sophistication', {})
        if comm_features:
            readability = comm_features.get('readability_score', 50)
            vocab_richness = comm_features.get('vocabulary_richness', 0)
            
            if readability > 70 and vocab_richness > 0.3:
                insights['communication_insight'] = "Demonstrates exceptional communication clarity with sophisticated vocabulary usage, indicating strong leadership and mentoring potential."
            elif readability > 50 and vocab_richness > 0.2:
                insights['communication_insight'] = "Shows solid technical communication skills with balanced complexity and accessibility."
            else:
                insights['communication_insight'] = "Communication style suggests focus on technical execution over documentation, common in specialized technical roles."
        
        # Technical depth insight
        tech_features = features.get('technical_depth', {})
        if tech_features:
            diversity = tech_features.get('language_diversity', 0)
            tech_adoption = tech_features.get('technology_adoption', 0)
            
            if diversity >= 5 and tech_adoption > 50:
                insights['technical_insight'] = "Polyglot programmer with strong technology adoption patterns, excellent for innovation-driven roles and technical leadership."
            elif diversity >= 3 and tech_adoption > 30:
                insights['technical_insight'] = "Well-rounded technical foundation with good learning agility and technology awareness."
            else:
                insights['technical_insight'] = "Focused technical specialization, ideal for deep expertise roles and domain-specific challenges."
        
        # Innovation pattern insight
        innovation_features = features.get('innovation_patterns', {})
        if innovation_features:
            velocity = innovation_features.get('innovation_velocity', 0)
            learning = innovation_features.get('learning_orientation', 0)
            
            if velocity > 40 and learning > 20:
                insights['innovation_insight'] = "High innovation velocity with strong learning orientation, perfect for fast-paced startups and R&D roles."
            elif velocity > 20 or learning > 10:
                insights['innovation_insight'] = "Balanced approach to innovation with steady learning patterns, suitable for growth-stage companies."
            else:
                insights['innovation_insight'] = "Stable, production-focused approach with emphasis on reliability over experimentation."
        
        return insights
