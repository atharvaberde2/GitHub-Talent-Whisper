#!/usr/bin/env python3
"""
GitHub Talent Whisperer - Development Server
Run this script to start the application locally.
"""

import os
from app import app

if __name__ == '__main__':
    # Set environment variables for development
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('FLASK_DEBUG', 'True')
    
    print("🚀 Starting GitHub Talent Whisperer...")
    print("📊 Backend: Flask + GitHub API")
    print("🎨 Frontend: HTML + Tailwind CSS")
    print("🔗 Access: http://localhost:5000")
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=True
    )
