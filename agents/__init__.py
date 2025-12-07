"""
BioGraphX Agent Package

This module exposes all agent classes used in the
AgentGraphPipeline. Each agent performs one step in the
end-to-end biomedical question-answering flow.

Agents:
    - QuestionAgent        → Extract biomedical entities (SciSpaCy)
    - NormalizeAgent       → Fuzzy normalization via entity_mappings.csv
    - WikipediaAgent       → Retrieve general medical knowledge from Wikipedia
    - RetrieverAgent       → Retrieve top PubMed evidence (ChromaDB)
    - QAModelAgent         → Generate answer using Qwen (ReAct prompting)
    - EvidenceAgent        → Format and validate evidence
    - ExplanationAgent     → Merge answer and evidence
"""


# __init__.py — allow easy imports for all agents

from .question_agent import QuestionAgent
from .normalize_agent import NormalizeAgent
from .wikipedia_agent import WikipediaAgent
# GraphAgent removed - Neo4j graph lacks entity-to-entity relationships
from .retriever_agent import RetrieverAgent
from .qa_model_agent import QAModelAgent
from .evidence_agent import EvidenceAgent
from .explanation_agent import ExplanationAgent
from .agent_graph_orchestrator import AgentGraphPipeline