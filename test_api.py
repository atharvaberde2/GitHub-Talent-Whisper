#!/usr/bin/env python3
"""
Test script for GitHub Talent Whisperer API
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing GitHub Talent Whisperer API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test analysis endpoint with a well-known GitHub user
    test_users = ["octocat", "torvalds", "gaearon"]
    
    for username in test_users:
        print(f"2. Testing analysis for @{username}...")
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/analyze/{username}")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Analysis completed in {end_time - start_time:.2f}s")
                print(f"   Username: {data.get('username')}")
                print(f"   Archetype: {data.get('archetype')}")
                print(f"   Overall Score: {data.get('overall_score')}")
                print(f"   Overall Confidence: {data.get('overall_confidence')}%")
                print(f"   Data Summary: {data.get('data_summary')}")
                
                # Show top talent
                talents = data.get('hidden_talents', {})
                if talents:
                    print("   Top Talents:")
                    for talent_name, talent_data in talents.items():
                        score = talent_data.get('score', 0)
                        print(f"     • {talent_name}: {score}/100")
                
                break  # Success, no need to test more users
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Analysis error: {e}")
        
        print()
    
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_api()
