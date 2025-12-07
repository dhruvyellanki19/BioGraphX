# app/main.py
# BioGraphX Flask Web Application
# Complete production-ready web interface for biomedical QA system

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pipeline_wrapper import PipelineWrapper
import os
import sys
import json
import traceback
from datetime import datetime
import logging

# Fix Python path for agent imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'biographx-production-key')
CORS(app)

# Global pipeline wrapper - loads once
pipeline_wrapper = None

def get_pipeline():
    """Get or initialize pipeline wrapper"""
    global pipeline_wrapper
    if pipeline_wrapper is None:
        try:
            logger.info(" Initializing BioGraphX pipeline...")
            pipeline_wrapper = PipelineWrapper()
            logger.info(" Pipeline ready for requests")
        except Exception as e:
            logger.error(f" Pipeline initialization failed: {e}")
            pipeline_wrapper = None
    return pipeline_wrapper

@app.route('/')
def index():
    """Main chat interface - ChatGPT style"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Process biomedical questions through the agent pipeline"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a valid question'}), 400
        
        logger.info(f"Processing question: {question}")
        start_time = datetime.now()
        
        # Get pipeline
        pipeline = get_pipeline()
        if not pipeline:
            return jsonify({'error': 'Pipeline not available'}), 500
        
        # Run pipeline
        result = pipeline.run_pipeline(question)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format response
        response = {
            'answer': result.get('answer', ''),
            'entities': result.get('entities', []),
            'normalized': result.get('normalized_entities', []),
            'triples': result.get('graph_triples', []),
            'evidence': result.get('evidence', []),
            'wikipedia': result.get('wikipedia_evidence', []),
            'explanation': result.get('explanation', ''),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Response generated in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing question: {traceback.format_exc()}")
        return jsonify({
            'error': f'Error processing question: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/graph')
def generate_graph():
    """Generate interactive graph visualization"""
    try:
        pipeline = get_pipeline()
        if not pipeline:
            return jsonify({'error': 'Pipeline not available'}), 500
        
        # Generate graph HTML
        graph_html = pipeline.generate_graph_html()
        return graph_html
        
    except Exception as e:
        logger.error(f" Error generating graph: {e}")
        return f"<html><body><h2>Graph Generation Error</h2><p>{str(e)}</p></body></html>"

@app.route('/health')
def health_check():
    """Health check endpoint"""
    pipeline = get_pipeline()
    return jsonify({
        'status': 'healthy' if pipeline else 'degraded',
        'pipeline_ready': pipeline is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    # Initialize pipeline on startup
    get_pipeline()
    
    # Run Flask app
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
@app.route('/test')
def test_page():
    """Simple test page for debugging"""
    return render_template('test.html')
