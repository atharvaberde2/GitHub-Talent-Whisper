#!/usr/bin/env python3
"""
🧠 TOURNAMENT-GRADE Psychological Profiler
Multi-dimensional personality and career aptitude analysis using behavioral psychology
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class PersonalityProfile:
    """Comprehensive personality profile based on Big Five + Technical traits"""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    technical_depth: float
    innovation_drive: float
    leadership_potential: float
    
class PsychologicalProfiler:
    """Advanced psychological profiling for developer talent assessment"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Research-backed personality-career correlations
        self.career_archetypes = {
            "Tech Innovation Visionary": {
                "openness": 85, "conscientiousness": 70, "extraversion": 65,
                "agreeableness": 60, "neuroticism": 40, "technical_depth": 90,
                "innovation_drive": 95, "leadership_potential": 80,
                "description": "Drives technological innovation with visionary thinking",
                "ideal_roles": ["CTO", "Technical Co-founder", "Innovation Lead"],
                "salary_premium": 0.35
            },
            "Systems Architecture Mastermind": {
                "openness": 75, "conscientiousness": 90, "extraversion": 45,
                "agreeableness": 55, "neuroticism": 30, "technical_depth": 95,
                "innovation_drive": 70, "leadership_potential": 70,
                "description": "Designs and builds scalable, robust technical systems",
                "ideal_roles": ["Principal Engineer", "Solution Architect", "Platform Lead"],
                "salary_premium": 0.30
            },
            "Technical Leadership Champion": {
                "openness": 70, "conscientiousness": 85, "extraversion": 80,
                "agreeableness": 85, "neuroticism": 35, "technical_depth": 80,
                "innovation_drive": 75, "leadership_potential": 95,
                "description": "Combines technical expertise with exceptional people leadership",
                "ideal_roles": ["Engineering Manager", "Tech Lead", "VP Engineering"],
                "salary_premium": 0.40
            },
            "Product Engineering Specialist": {
                "openness": 80, "conscientiousness": 75, "extraversion": 70,
                "agreeableness": 75, "neuroticism": 40, "technical_depth": 75,
                "innovation_drive": 85, "leadership_potential": 65,
                "description": "Bridges technical execution with product vision",
                "ideal_roles": ["Senior Product Engineer", "Technical Product Manager", "Full-stack Lead"],
                "salary_premium": 0.25
            },
            "Reliability Engineering Expert": {
                "openness": 60, "conscientiousness": 95, "extraversion": 50,
                "agreeableness": 70, "neuroticism": 25, "technical_depth": 85,
                "innovation_drive": 60, "leadership_potential": 60,
                "description": "Ensures system reliability and operational excellence",
                "ideal_roles": ["Site Reliability Engineer", "DevOps Lead", "Infrastructure Engineer"],
                "salary_premium": 0.20
            },
            "Research & Development Pioneer": {
                "openness": 95, "conscientiousness": 70, "extraversion": 55,
                "agreeableness": 60, "neuroticism": 45, "technical_depth": 90,
                "innovation_drive": 95, "leadership_potential": 50,
                "description": "Pushes the boundaries of what's technically possible",
                "ideal_roles": ["Research Scientist", "AI/ML Engineer", "Technical Researcher"],
                "salary_premium": 0.28
            },
            "Technical Mentorship Guide": {
                "openness": 75, "conscientiousness": 80, "extraversion": 75,
                "agreeableness": 90, "neuroticism": 30, "technical_depth": 80,
                "innovation_drive": 65, "leadership_potential": 85,
                "description": "Develops technical talent and builds engineering culture",
                "ideal_roles": ["Senior Engineer", "Technical Mentor", "Developer Advocate"],
                "salary_premium": 0.22
            }
        }
        
        # Career trajectory models
        self.trajectory_models = {
            "individual_contributor": {
                "technical_depth": 0.4, "innovation_drive": 0.3, "conscientiousness": 0.2, "openness": 0.1
            },
            "technical_leadership": {
                "leadership_potential": 0.4, "technical_depth": 0.3, "extraversion": 0.2, "agreeableness": 0.1
            },
            "product_leadership": {
                "leadership_potential": 0.3, "innovation_drive": 0.3, "extraversion": 0.2, "openness": 0.2
            },
            "executive_track": {
                "leadership_potential": 0.5, "extraversion": 0.3, "conscientiousness": 0.1, "agreeableness": 0.1
            }
        }
    
    def create_personality_profile(self, ml_features: Dict[str, Any]) -> PersonalityProfile:
        """Create comprehensive personality profile from ML features"""
        
        # Extract features
        comm_features = ml_features.get('communication_sophistication', {})
        tech_features = ml_features.get('technical_depth', {})
        behavioral_features = ml_features.get('behavioral_patterns', {})
        innovation_features = ml_features.get('innovation_patterns', {})
        commit_features = ml_features.get('commit_ml_features', {})
        
        # Map to Big Five personality traits
        openness = self._calculate_openness(innovation_features, tech_features, comm_features)
        conscientiousness = self._calculate_conscientiousness(commit_features, behavioral_features)
        extraversion = self._calculate_extraversion(behavioral_features, comm_features)
        agreeableness = self._calculate_agreeableness(behavioral_features, comm_features)
        neuroticism = self._calculate_neuroticism(commit_features, comm_features)
        
        # Technical traits
        technical_depth = self._calculate_technical_depth(tech_features)
        innovation_drive = self._calculate_innovation_drive(innovation_features)
        leadership_potential = self._calculate_leadership_potential(comm_features, behavioral_features)
        
        return PersonalityProfile(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
            technical_depth=technical_depth,
            innovation_drive=innovation_drive,
            leadership_potential=leadership_potential
        )
    
    def _calculate_openness(self, innovation_features: Dict, tech_features: Dict, comm_features: Dict) -> float:
        """Calculate Openness to Experience"""
        innovation_velocity = innovation_features.get('innovation_velocity', 0)
        language_diversity = tech_features.get('language_diversity', 0)
        vocab_richness = comm_features.get('vocabulary_richness', 0) * 100
        
        # Weighted combination
        openness = (innovation_velocity * 0.4 + 
                   min(100, language_diversity * 15) * 0.4 + 
                   vocab_richness * 0.2)
        
        return min(100, openness)
    
    def _calculate_conscientiousness(self, commit_features: Dict, behavioral_features: Dict) -> float:
        """Calculate Conscientiousness"""
        consistency = behavioral_features.get('consistency_score', 0)
        work_hour_ratio = commit_features.get('work_hour_ratio', 0.5) * 100
        frequency_consistency = commit_features.get('commit_frequency_consistency', 0.5) * 100
        
        conscientiousness = (consistency * 0.4 + 
                           work_hour_ratio * 0.3 + 
                           frequency_consistency * 0.3)
        
        return min(100, conscientiousness)
    
    def _calculate_extraversion(self, behavioral_features: Dict, comm_features: Dict) -> float:
        """Calculate Extraversion"""
        collaboration_tendency = behavioral_features.get('collaboration_tendency', 0)
        communication_volume = comm_features.get('communication_volume', 0)
        detailed_communication = behavioral_features.get('detailed_communication', 0)
        
        extraversion = (min(100, collaboration_tendency * 10) * 0.4 + 
                       min(100, communication_volume * 5) * 0.3 + 
                       detailed_communication * 0.3)
        
        return min(100, extraversion)
    
    def _calculate_agreeableness(self, behavioral_features: Dict, comm_features: Dict) -> float:
        """Calculate Agreeableness"""
        collaboration_tendency = behavioral_features.get('collaboration_tendency', 0)
        sentiment_consistency = comm_features.get('sentiment_consistency', 0.5) * 100
        psychological_traits = comm_features.get('psychological_traits', {})
        agreeableness_trait = psychological_traits.get('agreeableness', 0)
        
        agreeableness = (min(100, collaboration_tendency * 15) * 0.4 + 
                        sentiment_consistency * 0.3 + 
                        agreeableness_trait * 0.3)
        
        return min(100, agreeableness)
    
    def _calculate_neuroticism(self, commit_features: Dict, comm_features: Dict) -> float:
        """Calculate Neuroticism (emotional stability)"""
        sentiment_consistency = comm_features.get('sentiment_consistency', 0.5) * 100
        work_pattern_consistency = commit_features.get('commit_frequency_consistency', 0.5) * 100
        psychological_traits = comm_features.get('psychological_traits', {})
        neuroticism_trait = psychological_traits.get('neuroticism', 0)
        
        # Lower neuroticism = higher emotional stability
        emotional_stability = (sentiment_consistency * 0.4 + 
                              work_pattern_consistency * 0.3 + 
                              (100 - neuroticism_trait) * 0.3)
        
        neuroticism = 100 - emotional_stability  # Invert for neuroticism score
        return max(0, neuroticism)
    
    def _calculate_technical_depth(self, tech_features: Dict) -> float:
        """Calculate Technical Depth"""
        language_diversity = tech_features.get('language_diversity', 0)
        tech_adoption = tech_features.get('technology_adoption', 0)
        specialization = tech_features.get('specialization_focus', 0) * 100
        
        technical_depth = (min(100, language_diversity * 12) * 0.4 + 
                          tech_adoption * 0.4 + 
                          specialization * 0.2)
        
        return min(100, technical_depth)
    
    def _calculate_innovation_drive(self, innovation_features: Dict) -> float:
        """Calculate Innovation Drive"""
        innovation_velocity = innovation_features.get('innovation_velocity', 0)
        learning_orientation = innovation_features.get('learning_orientation', 0)
        experimentation = innovation_features.get('experimentation_score', 0)
        
        innovation_drive = (innovation_velocity * 0.4 + 
                           learning_orientation * 0.3 + 
                           experimentation * 0.3)
        
        return min(100, innovation_drive)
    
    def _calculate_leadership_potential(self, comm_features: Dict, behavioral_features: Dict) -> float:
        """Calculate Leadership Potential"""
        readability = comm_features.get('readability_score', 50)
        detailed_communication = behavioral_features.get('detailed_communication', 0)
        collaboration_tendency = behavioral_features.get('collaboration_tendency', 0)
        psychological_traits = comm_features.get('psychological_traits', {})
        
        # Leadership indicators
        communication_clarity = min(100, readability)
        mentoring_signals = detailed_communication
        team_orientation = min(100, collaboration_tendency * 10)
        
        leadership_potential = (communication_clarity * 0.3 + 
                               mentoring_signals * 0.4 + 
                               team_orientation * 0.3)
        
        return min(100, leadership_potential)
    
    def classify_archetype_advanced(self, profile: PersonalityProfile) -> Dict[str, Any]:
        """Advanced multi-dimensional archetype classification"""
        
        profile_vector = np.array([
            profile.openness, profile.conscientiousness, profile.extraversion,
            profile.agreeableness, profile.neuroticism, profile.technical_depth,
            profile.innovation_drive, profile.leadership_potential
        ])
        
        # Calculate similarity to each archetype
        archetype_scores = {}
        for archetype_name, archetype_traits in self.career_archetypes.items():
            archetype_vector = np.array([
                archetype_traits["openness"], archetype_traits["conscientiousness"],
                archetype_traits["extraversion"], archetype_traits["agreeableness"],
                archetype_traits["neuroticism"], archetype_traits["technical_depth"],
                archetype_traits["innovation_drive"], archetype_traits["leadership_potential"]
            ])
            
            # Calculate Euclidean distance (lower = better match)
            distance = euclidean_distances([profile_vector], [archetype_vector])[0][0]
            
            # Convert to similarity score (0-100)
            max_possible_distance = np.sqrt(8 * (100**2))  # 8 dimensions, max value 100
            similarity = max(0, 100 - (distance / max_possible_distance * 100))
            
            archetype_scores[archetype_name] = {
                "similarity": similarity,
                "description": archetype_traits["description"],
                "ideal_roles": archetype_traits["ideal_roles"],
                "salary_premium": archetype_traits["salary_premium"]
            }
        
        # Find best matches
        best_match = max(archetype_scores.items(), key=lambda x: x[1]["similarity"])
        
        # Calculate confidence based on how distinct the best match is
        sorted_scores = sorted(archetype_scores.values(), key=lambda x: x["similarity"], reverse=True)
        confidence = min(95, sorted_scores[0]["similarity"] + 
                        (sorted_scores[0]["similarity"] - sorted_scores[1]["similarity"]) * 0.5)
        
        # Determine secondary traits
        secondary_traits = []
        for archetype, scores in archetype_scores.items():
            if scores["similarity"] > 70 and archetype != best_match[0]:
                secondary_traits.append(archetype)
        
        # Calculate uniqueness score
        uniqueness = self._calculate_profile_uniqueness(profile_vector)
        
        return {
            "primary_archetype": best_match[0],
            "archetype_confidence": confidence,
            "archetype_details": best_match[1],
            "secondary_traits": secondary_traits[:2],  # Top 2 secondary traits
            "all_archetype_scores": archetype_scores,
            "profile_uniqueness": uniqueness,
            "personality_profile": profile
        }
    
    def _calculate_profile_uniqueness(self, profile_vector: np.ndarray) -> float:
        """Calculate how unique this profile is compared to typical patterns"""
        
        # Create typical profiles for comparison
        typical_profiles = [
            [50, 50, 50, 50, 50, 50, 50, 50],  # Average developer
            [30, 80, 30, 50, 40, 90, 40, 30],  # Typical backend specialist
            [70, 60, 70, 70, 40, 70, 80, 70],  # Typical full-stack developer
            [80, 70, 80, 80, 30, 80, 90, 90],  # Typical tech lead
        ]
        
        # Calculate distance from typical profiles
        distances = []
        for typical in typical_profiles:
            distance = euclidean_distances([profile_vector], [np.array(typical)])[0][0]
            distances.append(distance)
        
        # Higher average distance = more unique
        avg_distance = statistics.mean(distances)
        max_possible_distance = np.sqrt(8 * (100**2))
        if max_possible_distance > 0:
            uniqueness = min(100, (avg_distance / max_possible_distance) * 200)
        else:
            uniqueness = 50  # Default moderate uniqueness
        
        return uniqueness
    
    def predict_career_trajectory(self, profile: PersonalityProfile) -> Dict[str, Any]:
        """Predict optimal career trajectory based on personality profile"""
        
        profile_vector = np.array([
            profile.technical_depth, profile.innovation_drive, 
            profile.leadership_potential, profile.extraversion,
            profile.conscientiousness, profile.openness
        ])
        
        trajectory_scores = {}
        for trajectory, weights in self.trajectory_models.items():
            # Calculate weighted score for each trajectory
            score = 0
            if 'technical_depth' in weights:
                score += profile.technical_depth * weights['technical_depth']
            if 'innovation_drive' in weights:
                score += profile.innovation_drive * weights['innovation_drive']
            if 'leadership_potential' in weights:
                score += profile.leadership_potential * weights['leadership_potential']
            if 'extraversion' in weights:
                score += profile.extraversion * weights['extraversion']
            if 'conscientiousness' in weights:
                score += profile.conscientiousness * weights['conscientiousness']
            if 'openness' in weights:
                score += profile.openness * weights['openness']
            
            trajectory_scores[trajectory] = score
        
        # Normalize scores to 0-100
        max_score = max(trajectory_scores.values())
        if max_score > 0:
            trajectory_scores = {k: (v/max_score)*100 for k, v in trajectory_scores.items()}
        
        return {
            "trajectory_scores": trajectory_scores,
            "recommended_path": max(trajectory_scores.items(), key=lambda x: x[1]),
            "career_flexibility": statistics.stdev(list(trajectory_scores.values()))  # Lower = more specialized
        }
    
    def generate_career_recommendations(self, archetype_result: Dict[str, Any], 
                                      trajectory_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific career recommendations with salary impact"""
        
        primary_archetype = archetype_result["primary_archetype"]
        archetype_details = archetype_result["archetype_details"]
        trajectory_scores = trajectory_result["trajectory_scores"]
        
        recommendations = []
        
        # Primary recommendation based on archetype
        for i, role in enumerate(archetype_details["ideal_roles"][:3]):
            base_salary_impact = archetype_details["salary_premium"]
            
            # Adjust based on trajectory fit
            trajectory_bonus = 0
            if i == 0:  # First role gets trajectory bonus
                recommended_trajectory = trajectory_result["recommended_path"][0]
                trajectory_bonus = trajectory_result["trajectory_scores"][recommended_trajectory] / 500  # 0-0.2 bonus
            
            total_salary_impact = base_salary_impact + trajectory_bonus
            
            recommendations.append({
                "role": role,
                "match_percentage": int(archetype_result["archetype_confidence"] - (i * 5)),
                "salary_impact": f"+${int(total_salary_impact * 100)}k average increase",
                "reasoning": f"Perfect fit for {primary_archetype.lower()} with {archetype_details['description'].lower()}",
                "confidence": "high" if i == 0 else "medium"
            })
        
        return recommendations
