# app/pipeline_wrapper.py
# Wrapper for BioGraphX Agent Pipeline with graph visualization
# Handles pipeline initialization and caching

import sys
import os
import traceback
from pathlib import Path

# Fix Python path for agent imports
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from pyvis.network import Network
    import networkx as nx
except ImportError:
    print("PyVis not installed. Graph visualization disabled.")
    Network = None
    nx = None

class PipelineWrapper:
    """Wrapper for AgentGraphPipeline with additional web features"""
    
    def __init__(self):
        self.pipeline = None
        self.graph_cache = {}
        self.init_pipeline()
    
    def init_pipeline(self):
        """Initialize the agent pipeline"""
        try:
            from agents.agent_graph_orchestrator import AgentGraphPipeline
            print("Loading BioGraphX Agent Pipeline...")
            self.pipeline = AgentGraphPipeline()
            print("Pipeline loaded successfully!")
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            print(traceback.format_exc())
            raise e
    
    def run_pipeline(self, question):
        """Run the full agent pipeline and return results"""
        if not self.pipeline:
            raise Exception("Pipeline not initialized")
        
        try:
            # Run the 7-agent pipeline
            result = self.pipeline.run(question)
            
            # Process and clean results for web display
            processed_result = self.process_pipeline_output(result)
            
            return processed_result
            
        except Exception as e:
            print(f"Pipeline execution error: {e}")
            raise e
    
    def process_pipeline_output(self, result):
        """Process pipeline output for web display"""
        # Ensure all expected keys exist
        processed = {
            'question': result.get('question', ''),
            'entities': result.get('entities', []),
            'normalized_entities': result.get('normalized_entities', []),
            'graph_triples': result.get('graph_triples', []),
            'evidence': result.get('evidence', []),
            'wikipedia_evidence': result.get('wikipedia_evidence', []),
            'answer': result.get('answer', ''),
            'explanation': result.get('explanation', ''),
            'formatted_evidence': result.get('formatted_evidence', [])
        }
        
        # Clean and format triples for display
        if processed['graph_triples']:
            processed['graph_triples'] = self.format_triples(processed['graph_triples'])
        
        # Format evidence for display
        if processed['evidence']:
            processed['evidence'] = self.format_evidence(processed['evidence'])
        
        return processed
    
    def format_triples(self, triples):
        """Format graph triples for display"""
        formatted = []
        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                formatted.append({
                    'head': str(head),
                    'relation': str(relation),
                    'tail': str(tail)
                })
            elif isinstance(triple, dict):
                formatted.append({
                    'head': str(triple.get('head', triple.get('entity', ''))),
                    'relation': str(triple.get('relation', triple.get('rel_type', ''))),
                    'tail': str(triple.get('tail', triple.get('neighbor', '')))
                })
        return formatted
    
    def format_evidence(self, evidence):
        """Format evidence for display"""
        formatted = []
        for ev in evidence:
            if isinstance(ev, dict):
                formatted.append({
                    'pmid': str(ev.get('pmid', 'Unknown')),
                    'text': str(ev.get('sentence', ev.get('text', ''))),
                    'source': 'PubMed'
                })
            else:
                formatted.append({
                    'pmid': 'Unknown',
                    'text': str(ev),
                    'source': 'PubMed'
                })
        return formatted
    
    def generate_graph_html(self, triples=None, width="100%", height="400px"):
        """Generate interactive graph visualization using PyVis"""
        if not Network or not triples:
            return self.generate_fallback_graph()
        
        try:
            # Create network
            net = Network(
                width=width,
                height=height,
                bgcolor="#1a1a1a",  # Dark background
                font_color="#ffffff",
                directed=True
            )
            
            # Configure physics
            net.set_options("""
            var options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100}
                },
                "nodes": {
                    "borderWidth": 2,
                    "size": 20,
                    "color": {
                        "border": "#4a90e2",
                        "background": "#2c3e50"
                    },
                    "font": {"color": "#ffffff", "size": 12}
                },
                "edges": {
                    "color": {"color": "#7f8c8d"},
                    "width": 2,
                    "arrows": {"to": {"enabled": true, "scaleFactor": 1}}
                }
            }
            """)
            
            # Add nodes and edges
            nodes_added = set()
            for triple in triples:
                head = triple['head']
                tail = triple['tail']
                relation = triple['relation']
                
                # Add nodes if not already added
                if head not in nodes_added:
                    net.add_node(head, label=head, title=head)
                    nodes_added.add(head)
                
                if tail not in nodes_added:
                    net.add_node(tail, label=tail, title=tail)
                    nodes_added.add(tail)
                
                # Add edge
                net.add_edge(head, tail, label=relation, title=relation)
            
            # Generate HTML
            graph_path = os.path.join(os.path.dirname(__file__), 'static', 'graph')
            os.makedirs(graph_path, exist_ok=True)
            
            html_file = os.path.join(graph_path, 'graph.html')
            net.save_graph(html_file)
            
            # Return the HTML content
            with open(html_file, 'r') as f:
                return f.read()
                
        except Exception as e:
            print(f" Graph generation error: {e}")
            return self.generate_fallback_graph()
    
    def generate_fallback_graph(self):
        """Generate fallback graph when PyVis is not available"""
        return """
        <html>
        <head>
            <title>Graph Visualization</title>
            <style>
                body { 
                    background: #1a1a1a; 
                    color: white; 
                    font-family: Arial, sans-serif; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    height: 100vh; 
                    margin: 0; 
                }
                .message {
                    text-align: center;
                    padding: 20px;
                    border: 1px solid #444;
                    border-radius: 8px;
                    background: #2a2a2a;
                }
            </style>
        </head>
        <body>
            <div class="message">
                <h3>Graph Visualization</h3>
                <p>Interactive graph visualization will appear here when knowledge graph relationships are found.</p>
                <p><small>Powered by PyVis & NetworkX</small></p>
            </div>
        </body>
        </html>
        """
    
    def get_cached_graph(self, question):
        """Get cached graph for question if available"""
        return self.graph_cache.get(question)
    
    def cache_graph(self, question, graph_html):
        """Cache graph HTML for question"""
        self.graph_cache[question] = graph_html