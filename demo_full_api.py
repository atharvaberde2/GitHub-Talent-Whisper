#!/usr/bin/env python3
"""
Demo script for all GitHub Talent Whisperer API endpoints
Tests the complete API suite including batch analysis and pattern discovery
"""

import requests
import json
import time

def demo_complete_api():
    """Demo all API endpoints for judge preparation"""
    
    base_url = "http://localhost:5000"
    
    print("🚀 GitHub Talent Whisperer - Complete API Demo")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1. 🏥 Health Check")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Service: {health_data['service']}")
            print(f"   ✅ Status: {health_data['status']}")
            print(f"   📊 Cached Analyses: {health_data['cached_analyses']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    # 2. Single Analysis
    print("\n2. 🔍 Single User Analysis")
    test_user = "octocat"
    try:
        response = requests.get(f"{base_url}/api/analyze/{test_user}")
        if response.status_code == 200:
            analysis = response.json()
            print(f"   ✅ Analyzed: @{analysis['username']}")
            print(f"   🎯 Archetype: {analysis['archetype']}")
            print(f"   📊 Overall Score: {analysis['overall_score']}/100")
            print(f"   🆔 Analysis ID: {analysis.get('analysis_id', 'N/A')}")
            
            # Store for later tests
            analysis_id = analysis.get('analysis_id')
        else:
            print(f"   ❌ Analysis failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return
    
    # 3. Batch Analysis (Demo Prep)
    print("\n3. 📦 Batch Analysis for Demo Prep")
    judge_usernames = ["torvalds", "gaearon", "octocat"]  # Mock judge profiles
    
    try:
        batch_payload = {"usernames": judge_usernames}
        response = requests.post(f"{base_url}/api/batch-analyze", 
                               json=batch_payload,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            batch_result = response.json()
            print(f"   ✅ Batch ID: {batch_result['batch_id']}")
            print(f"   📊 Status: {batch_result['status']}")
            print(f"   ✅ Successful: {batch_result['summary']['successful']}")
            print(f"   ❌ Failed: {batch_result['summary']['failed']}")
            
            # Show sample insights
            if batch_result['results']:
                sample_user = list(batch_result['results'].keys())[0]
                sample_result = batch_result['results'][sample_user]
                print(f"   🎯 Sample - {sample_user}: {sample_result.get('archetype', 'Unknown')}")
                
        else:
            print(f"   ❌ Batch analysis failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Batch analysis error: {e}")
    
    # 4. Pattern Discovery
    print("\n4. 🔍 Pattern Discovery")
    try:
        response = requests.get(f"{base_url}/api/patterns/discovery")
        if response.status_code == 200:
            patterns = response.json()
            print(f"   ✅ Profiles Analyzed: {patterns['profiles_analyzed']}")
            
            # Show archetype distribution
            if 'discovered_patterns' in patterns:
                archetypes = patterns['discovered_patterns'].get('archetype_distribution', {})
                print(f"   📊 Archetype Distribution:")
                for archetype, count in archetypes.items():
                    print(f"      • {archetype}: {count}")
                
                # Show leadership clusters
                leadership = patterns['discovered_patterns'].get('leadership_clusters', [])
                if leadership:
                    cluster = leadership[0]
                    print(f"   👥 Leadership Pattern: {cluster['description']}")
                    print(f"      Average Score: {cluster['average_score']}")
                
        else:
            print(f"   ❌ Pattern discovery failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Pattern discovery error: {e}")
    
    # 5. Cached Insights Retrieval
    print("\n5. 💾 Cached Insights Retrieval")
    if analysis_id:
        try:
            response = requests.get(f"{base_url}/api/insights/{analysis_id}")
            if response.status_code == 200:
                cached_result = response.json()
                print(f"   ✅ Retrieved cached analysis for: @{cached_result['username']}")
                print(f"   🕒 Retrieved at: {cached_result.get('retrieved_at', 'N/A')}")
                print(f"   💾 Cache hit: {cached_result.get('cache_hit', False)}")
            else:
                print(f"   ❌ Cached retrieval failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Cached retrieval error: {e}")
    
    # 6. Profile Type Recommendations
    print("\n6. 🎯 Profile Type Recommendations")
    profile_types = ["async_leadership", "innovation_hunter", "knowledge_transfer"]
    
    for profile_type in profile_types:
        try:
            response = requests.get(f"{base_url}/api/recommendations/{profile_type}")
            if response.status_code == 200:
                recs = response.json()
                print(f"   ✅ {profile_type}:")
                for rec in recs['recommendations'][:2]:  # Show first 2
                    print(f"      • {rec['role']} (Match: {rec['match']}%)")
                    print(f"        💰 {rec['salary_impact']}")
            else:
                print(f"   ❌ Recommendations failed for {profile_type}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Recommendations error for {profile_type}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Complete API Demo Finished!")
    print("\n🎪 Judge Demo Preparation Features:")
    print("✅ Batch analysis for pre-analyzing judge profiles")
    print("✅ Pattern discovery across multiple developers")
    print("✅ Cached insights for instant demo responses")
    print("✅ Profile-type specific recommendations")
    print("✅ Real-time health monitoring")
    
    print("\n🚀 Ready for Nuclear Demo Strategy!")

def demo_judge_preparation():
    """Demonstrate the judge preparation workflow"""
    
    print("\n" + "🎪" * 20)
    print("JUDGE PREPARATION WORKFLOW DEMO")
    print("🎪" * 20)
    
    # Simulate preparing for judges
    mock_judges = {
        "judge1": "torvalds",  # Systems programming expert
        "judge2": "gaearon",   # React ecosystem leader  
        "judge3": "octocat"    # GitHub platform
    }
    
    base_url = "http://localhost:5000"
    
    print(f"\n📋 Preparing analysis for {len(mock_judges)} judges...")
    
    # Pre-analyze all judges
    usernames = list(mock_judges.values())
    batch_payload = {"usernames": usernames}
    
    try:
        response = requests.post(f"{base_url}/api/batch-analyze", 
                               json=batch_payload,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Pre-analysis complete! Batch ID: {results['batch_id']}")
            
            # Generate insights for each judge
            for judge_role, username in mock_judges.items():
                if username in results['results']:
                    analysis = results['results'][username]
                    archetype = analysis.get('archetype', 'Unknown')
                    score = analysis.get('overall_score', 0)
                    
                    print(f"\n🎯 {judge_role.upper()} (@{username}):")
                    print(f"   Archetype: {archetype}")
                    print(f"   Overall Score: {score}/100")
                    
                    # Show top talent
                    talents = analysis.get('hidden_talents', {})
                    top_talent = max(talents.items(), key=lambda x: x[1].get('score', 0)) if talents else None
                    if top_talent:
                        talent_name, talent_data = top_talent
                        print(f"   Top Strength: {talent_name.replace('_', ' ').title()} ({talent_data.get('score', 0)}/100)")
                        print(f"   Evidence: {talent_data.get('evidence', 'N/A')}")
            
            print(f"\n🎪 Demo Strategy:")
            print(f"1. 'Let me analyze your GitHub live...'")
            print(f"2. *Shows pre-computed insights instantly*")
            print(f"3. 'Fascinating! Your {archetype} pattern suggests...'")
            print(f"4. *Reveals surprising career insights*")
            print(f"5. 'This is why traditional recruiting misses talent like you!'")
            
        else:
            print(f"❌ Judge preparation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Judge preparation error: {e}")

if __name__ == "__main__":
    demo_complete_api()
    demo_judge_preparation()
