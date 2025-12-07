from agents.question_agent import QuestionAgent
from agents.normalize_agent import NormalizeAgent
from agents.wikipedia_agent import WikipediaAgent
from agents.retriever_agent import RetrieverAgent
from agents.qa_model_agent import QAModelAgent
from agents.evidence_agent import EvidenceAgent
from agents.explanation_agent import ExplanationAgent


class AgentGraphPipeline:
    def __init__(self):
        print("[Pipeline] Initializing agents...\n")

        # Load models ONCE
        self.q = QuestionAgent()
        self.n = NormalizeAgent()
        self.w = WikipediaAgent()  # Wikipedia for general medical knowledge
        self.r = RetrieverAgent()
        self.m = QAModelAgent()
        self.e = EvidenceAgent()
        self.x = ExplanationAgent()

    def run(self, question):
        state = {"question": question}

        # All agents are stateless → safe to reuse
        # WikipediaAgent added for complementary knowledge
        for agent in [self.q, self.n, self.w, self.r, self.m, self.e, self.x]:
            state = agent.run(state)

        return state
