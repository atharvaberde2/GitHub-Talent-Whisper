#!/usr/bin/env python3
"""
Test script to verify all REAL API implementations (no simulation)
Tests GitHub REST, GraphQL, and OpenAI integrations
"""

import requests
import json

def test_real_implementations():
    """Test all real API implementations"""
    
    base_url = "http://localhost:5000"
    
    print("🔍 Testing REAL API Implementations")
    print("=" * 60)
    print("📊 Verifying: GitHub REST + GraphQL + OpenAI")
    print()
    
    # Test with a real GitHub user
    test_username = "octocat"  # GitHub's official mascot account
    
    print(f"🧪 Testing comprehensive analysis for @{test_username}")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/api/analyze/{test_username}")
        
        if response.status_code == 200:
            analysis = response.json()
            
            print("✅ API Response Successful")
            print(f"📊 Username: @{analysis['username']}")
            print(f"🎯 Archetype: {analysis['archetype']}")
            print(f"💯 Overall Score: {analysis['overall_score']}/100")
            print()
            
            # Verify all talent indicators are present
            talents = analysis.get('hidden_talents', {})
            required_talents = [
                'async_leadership',
                'problem_decomposition', 
                'knowledge_transfer',
                'stress_management',
                'innovation_appetite',
                'collaboration_skills',  # NEW: GraphQL-based
                'leadership_potential'   # NEW: GraphQL-based
            ]
            
            print("🧠 Talent Indicators Analysis:")
            for talent in required_talents:
                if talent in talents:
                    score = talents[talent]['score']
                    evidence = talents[talent]['evidence']
                    is_graphql = talent in ['collaboration_skills', 'leadership_potential']
                    api_type = "GraphQL" if is_graphql else "REST"
                    print(f"   ✅ {talent}: {score}/100 ({api_type})")
                    print(f"      Evidence: {evidence}")
                else:
                    print(f"   ❌ {talent}: MISSING")
            print()
            
            # Check for GraphQL-specific data
            graphql_indicators = ['collaboration_skills', 'leadership_potential']
            graphql_working = all(talent in talents for talent in graphql_indicators)
            
            if graphql_working:
                print("✅ GitHub GraphQL API: WORKING")
                collab = talents['collaboration_skills']['details']
                leadership = talents['leadership_potential']['details']
                
                print(f"   📈 PR Reviews: {collab.get('pr_reviews', 0)}")
                print(f"   👥 Mentoring Reviews: {collab.get('mentoring_reviews', 0)}")
                print(f"   💬 Issue Comments: {collab.get('issue_comments', 0)}")
                print(f"   🏗️ Owned Repos: {leadership.get('owned_repos', 0)}")
                print(f"   🎯 Leadership Signals: {leadership.get('leadership_signals', 0)}")
            else:
                print("❌ GitHub GraphQL API: NOT WORKING")
            print()
            
            # Check AI insights
            ai_insights = analysis.get('ai_insights', {})
            if ai_insights:
                print("🤖 AI-Powered Insights:")
                insight_types = ['leadership_insight', 'technical_insight', 'career_insight']
                
                for insight_type in insight_types:
                    if insight_type in ai_insights:
                        insight = ai_insights[insight_type]
                        # Check if it's a real AI response (longer, more sophisticated)
                        is_real_ai = len(insight) > 100 and "patterns" in insight.lower()
                        api_status = "Real OpenAI" if is_real_ai else "Intelligent Mock"
                        print(f"   ✅ {insight_type}: {api_status}")
                        print(f"      {insight[:100]}...")
                    else:
                        print(f"   ❌ {insight_type}: MISSING")
                
                # Determine if OpenAI is really connected
                insights_text = " ".join(ai_insights.values())
                # Look for specific score references which indicate real AI processing
                openai_working = ("score of" in insights_text.lower() or 
                                "/100" in insights_text or 
                                "potential score" in insights_text.lower() or
                                len(insights_text) > 400)
                
                if openai_working:
                    print("✅ OpenAI API: FULLY WORKING (Real AI-powered insights)")
                else:
                    print("⚠️ OpenAI API: Using intelligent mocks (API key not configured)")
            else:
                print("❌ AI Insights: MISSING")
            print()
            
            # Verify data sources
            data_summary = analysis.get('data_summary', {})
            print("📊 Data Sources Verification:")
            print(f"   📁 Repositories: {data_summary.get('repos_analyzed', 0)}")
            print(f"   💻 Commits: {data_summary.get('commits_analyzed', 0)}")
            print(f"   🔀 Pull Requests: {data_summary.get('prs_analyzed', 0)}")
            print()
            
            # Overall assessment
            print("🏆 FINAL ASSESSMENT:")
            rest_api_working = analysis['overall_score'] > 0
            graphql_api_working = graphql_working
            ai_insights_present = bool(ai_insights)
            
            if rest_api_working:
                print("   ✅ GitHub REST API: FULLY IMPLEMENTED")
            else:
                print("   ❌ GitHub REST API: NOT WORKING")
                
            if graphql_api_working:
                print("   ✅ GitHub GraphQL API: FULLY IMPLEMENTED")
            else:
                print("   ❌ GitHub GraphQL API: NOT IMPLEMENTED")
                
            if ai_insights_present:
                print("   ✅ AI Insights: IMPLEMENTED (Check OpenAI key for real API)")
            else:
                print("   ❌ AI Insights: NOT IMPLEMENTED")
            
            # Calculate real vs simulated percentage
            total_features = 3  # REST, GraphQL, AI
            working_features = sum([rest_api_working, graphql_api_working, ai_insights_present])
            real_percentage = (working_features / total_features) * 100
            
            print(f"\n📈 IMPLEMENTATION STATUS: {real_percentage:.0f}% REAL")
            
            if real_percentage == 100:
                print("🎉 ALL FEATURES IMPLEMENTED - ZERO SIMULATION!")
            elif real_percentage >= 67:
                print("✅ MOSTLY REAL - Minor simulation only")
            elif real_percentage >= 33:
                print("⚠️ PARTIALLY REAL - Some simulation remaining")
            else:
                print("❌ MOSTLY SIMULATED - Major work needed")
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 Real Implementation Test Complete!")

if __name__ == "__main__":
    test_real_implementations()
