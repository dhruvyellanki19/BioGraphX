# BioGraphX Flask Application Runner

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app from main.py
from main import app

if __name__ == '__main__':
    # Configuration for different environments
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 8989))  # Using port 8989
    host = os.getenv('HOST', '127.0.0.1')  # Changed to localhost for better security
    
    print(f"Starting BioGraphX Flask Application...")
    print(f"Debug Mode: {debug_mode}")
    print(f"Host: {host}:{port}")
    print(f"Access URL: http://localhost:{port}")
    print(f"Press Ctrl+C to stop the server")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True
        )
    except KeyboardInterrupt:
        print(f"\n Server stopped by user")
    except Exception as e:
        print(f" Error starting server: {e}")
        sys.exit(1)