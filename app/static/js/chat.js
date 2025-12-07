// BioGraphX Chat Interface JavaScript

class ChatInterface {
    constructor() {
        this.isProcessing = false;
        this.lastResponse = null;
        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.setupSectionToggle();
        this.setupExampleButtons();
        this.focusInput();
    }

    setupElements() {
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.panelContent = document.querySelector('.panel-content');
        this.characterCount = document.querySelector('.character-count');
        this.loadingOverlay = document.querySelector('.loading-overlay');
    }

    setupEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Input events
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.chatInput.addEventListener('input', () => {
            this.updateCharacterCount();
            this.autoResizeInput();
        });

        // Panel toggle
        const panelToggle = document.querySelector('.panel-toggle');
        if (panelToggle) {
            panelToggle.addEventListener('click', () => this.togglePanel());
        }
    }

    setupSectionToggle() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('.section-header')) {
                const header = e.target.closest('.section-header');
                const content = header.nextElementSibling;

                if (content && content.classList.contains('section-content')) {
                    header.classList.toggle('collapsed');
                    content.style.display = header.classList.contains('collapsed') ? 'none' : 'block';
                }
            }
        });
    }

    setupExampleButtons() {
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.chatInput.value = btn.textContent;
                this.focusInput();
                this.updateCharacterCount();
            });
        });
    }

    updateCharacterCount() {
        if (this.characterCount) {
            const length = this.chatInput.value.length;
            this.characterCount.textContent = `${length}/2000`;

            if (length > 1800) {
                this.characterCount.style.color = '#dc3545';
            } else if (length > 1500) {
                this.characterCount.style.color = '#ffc107';
            } else {
                this.characterCount.style.color = '#666';
            }
        }
    }

    autoResizeInput() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
    }

    focusInput() {
        setTimeout(() => {
            this.chatInput.focus();
        }, 100);
    }

    togglePanel() {
        const panel = document.querySelector('.agent-panel');
        panel.classList.toggle('collapsed');

        const toggle = document.querySelector('.panel-toggle i');
        if (panel.classList.contains('collapsed')) {
            toggle.className = 'fas fa-chevron-left';
        } else {
            toggle.className = 'fas fa-chevron-right';
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;

        // Disable input and show processing state
        this.setProcessingState(true);

        // Add user message to chat
        this.addMessage(message, 'user');

        // Clear input
        this.chatInput.value = '';
        this.updateCharacterCount();
        this.autoResizeInput();

        try {
            // Show processing indicator in panel
            this.showProcessingIndicator();

            // Send request to backend
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Store response for panel display
            this.lastResponse = data;

            // Add assistant response to chat
            this.addMessage(data.answer, 'assistant');

            // Update agent panel
            this.updateAgentPanel(data);

        } catch (error) {
            console.error('Error sending message:', error);
            this.showError(error.message);
            this.hideProcessingIndicator();
        } finally {
            this.setProcessingState(false);
        }
    }

    setProcessingState(processing) {
        this.isProcessing = processing;
        this.sendButton.disabled = processing;
        this.chatInput.disabled = processing;

        if (processing) {
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }

    addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const icon = document.createElement('div');
        icon.className = `${role}-icon`;
        icon.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = this.formatMessageContent(content);

        messageContent.appendChild(icon);
        messageContent.appendChild(messageText);
        messageDiv.appendChild(messageContent);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessageContent(content) {
        // Convert newlines to <br> and preserve formatting
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^(.*)$/, '<p>$1</p>')
            .replace(/<p><\/p>/g, '')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    showProcessingIndicator() {
        const placeholder = this.panelContent.querySelector('.panel-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }

        const processingDiv = document.createElement('div');
        processingDiv.className = 'processing-indicator';
        processingDiv.innerHTML = `
            <div class="processing-header">
                <i class="fas fa-cogs fa-spin"></i>
                <span>Processing your question...</span>
            </div>
            <div class="agent-steps">
                <div class="agent-step active" data-step="question">
                    <i class="fas fa-question-circle"></i>
                    <span>Analyzing question</span>
                </div>
                <div class="agent-step" data-step="wikipedia">
                    <i class="fas fa-wikipedia-w"></i>
                    <span>Fetching medical knowledge</span>
                </div>
                <div class="agent-step" data-step="retrieval">
                    <i class="fas fa-search"></i>
                    <span>Retrieving evidence</span>
                </div>
                <div class="agent-step" data-step="generation">
                    <i class="fas fa-brain"></i>
                    <span>Generating answer</span>
                </div>
                <div class="agent-step" data-step="explanation">
                    <i class="fas fa-lightbulb"></i>
                    <span>Creating explanation</span>
                </div>
            </div>
        `;

        this.panelContent.innerHTML = '';
        this.panelContent.appendChild(processingDiv);

        // Simulate step progression
        this.simulateProcessingSteps();
    }

    simulateProcessingSteps() {
        const steps = ['question', 'wikipedia', 'retrieval', 'generation', 'explanation'];
        let currentStep = 0;

        const progressTimer = setInterval(() => {
            if (currentStep < steps.length) {
                // Mark current step as completed
                const currentStepElement = document.querySelector(`[data-step="${steps[currentStep]}"]`);
                if (currentStepElement) {
                    currentStepElement.classList.remove('active');
                    currentStepElement.classList.add('completed');
                    currentStepElement.querySelector('i').className = 'fas fa-check-circle';
                }

                currentStep++;

                // Activate next step
                if (currentStep < steps.length) {
                    const nextStepElement = document.querySelector(`[data-step="${steps[currentStep]}"]`);
                    if (nextStepElement) {
                        nextStepElement.classList.add('active');
                    }
                }
            } else {
                clearInterval(progressTimer);
            }
        }, 800);

        // Store timer reference for cleanup
        this.progressTimer = progressTimer;
    }

    hideProcessingIndicator() {
        if (this.progressTimer) {
            clearInterval(this.progressTimer);
            this.progressTimer = null;
        }
    }

    updateAgentPanel(data) {
        this.hideProcessingIndicator();

        const agentOutputs = document.createElement('div');
        agentOutputs.className = 'agent-outputs';

        // Entities Section
        if (data.entities && data.entities.length > 0) {
            agentOutputs.appendChild(this.createEntitiesSection(data.entities));
        }

        // Wikipedia Section
        if (data.wikipedia && data.wikipedia.length > 0) {
            agentOutputs.appendChild(this.createWikipediaSection(data.wikipedia));
        }

        // Graph Triples Section
        if (data.triples && data.triples.length > 0) {
            agentOutputs.appendChild(this.createGraphSection(data.triples, data.graph_html));
        }

        // Evidence Section
        if (data.evidence && data.evidence.length > 0) {
            agentOutputs.appendChild(this.createEvidenceSection(data.evidence));
        }

        // Explanation Section
        if (data.explanation) {
            agentOutputs.appendChild(this.createExplanationSection(data.explanation));
        }

        this.panelContent.innerHTML = '';
        this.panelContent.appendChild(agentOutputs);
    }

    createEntitiesSection(entities) {
        const section = document.createElement('div');
        section.className = 'output-section';

        section.innerHTML = `
            <div class="section-header">
                <div>
                    <i class="fas fa-tags"></i>
                    <span>Extracted Entities</span>
                </div>
                <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="section-content">
                <div class="entities-grid">
                    ${entities.map(entity =>
            `<span class="entity-tag">${this.escapeHtml(entity)}</span>`
        ).join('')}
                </div>
            </div>
        `;

        return section;
    }

    createWikipediaSection(wikipedia) {
        const section = document.createElement('div');
        section.className = 'output-section';

        section.innerHTML = `
            <div class="section-header">
                <div>
                    <i class="fas fa-wikipedia-w"></i>
                    <span>Wikipedia Knowledge (${wikipedia.length} articles)</span>
                </div>
                <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="section-content">
                <div class="wikipedia-list">
                    ${wikipedia.map(article => `
                        <div class="wikipedia-item">
                            <div class="wikipedia-header">
                                <h4>${this.escapeHtml(article.title)}</h4>
                                <a href="${article.url}" target="_blank" class="wiki-link">
                                    <i class="fas fa-external-link-alt"></i> View on Wikipedia
                                </a>
                            </div>
                            <div class="wikipedia-summary">${this.escapeHtml(article.summary)}</div>
                            <div class="wikipedia-entity">
                                <i class="fas fa-tag"></i> Related to: ${this.escapeHtml(article.entity)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        return section;
    }

    createGraphSection(triples, graphHtml) {
        const section = document.createElement('div');
        section.className = 'output-section';

        let graphContent = '';
        if (graphHtml) {
            graphContent = `
                <div class="graph-viz">
                    <iframe id="graphFrame" srcdoc="${this.escapeHtml(graphHtml)}"></iframe>
                </div>
            `;
        } else {
            graphContent = `
                <div class="graph-viz">
                    <div class="graph-placeholder">
                        <i class="fas fa-project-diagram"></i>
                        <p>No graph visualization available</p>
                    </div>
                </div>
            `;
        }

        section.innerHTML = `
            <div class="section-header">
                <div>
                    <i class="fas fa-book-medical"></i>
                    <span>Medical Knowledge & Context</span>
                </div>
                <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="section-content">
                <div class="triples-container">
                    ${graphContent}
                    <div class="triples-table">
                        ${triples.map(triple => `
                            <div class="triple-row">
                                <div class="triple-text">
                                    <span class="triple-entity">${this.escapeHtml(triple.subject || triple[0])}</span>
                                    <span class="triple-relation">${this.escapeHtml(triple.predicate || triple[1])}</span>
                                    <span class="triple-entity">${this.escapeHtml(triple.object || triple[2])}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        return section;
    }

    createEvidenceSection(evidence) {
        const section = document.createElement('div');
        section.className = 'output-section';

        section.innerHTML = `
            <div class="section-header">
                <div>
                    <i class="fas fa-file-text"></i>
                    <span>Evidence (${evidence.length} articles)</span>
                </div>
                <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="section-content">
                <div class="evidence-list">
                    ${evidence.map(item => `
                        <div class="evidence-item">
                            <div class="evidence-header">
                                <span>Evidence</span>
                                ${item.pmid ? `<span class="pmid-badge">PMID: ${item.pmid}</span>` : ''}
                            </div>
                            <div class="evidence-text">${this.escapeHtml(item.text || item)}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        return section;
    }

    createExplanationSection(explanation) {
        const section = document.createElement('div');
        section.className = 'output-section';

        section.innerHTML = `
            <div class="section-header">
                <div>
                    <i class="fas fa-lightbulb"></i>
                    <span>Reasoning Explanation</span>
                </div>
                <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="section-content">
                <div class="explanation-text">${this.formatMessageContent(explanation)}</div>
            </div>
        `;

        return section;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        const errorModal = document.createElement('div');
        errorModal.className = 'error-modal';

        errorModal.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <span>
                        <i class="fas fa-exclamation-triangle"></i>
                        Error
                    </span>
                    <button class="close-button" onclick="this.closest('.error-modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="error-body">
                    <p>Sorry, there was an error processing your request:</p>
                    <p><strong>${this.escapeHtml(message)}</strong></p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>
            </div>
        `;

        document.body.appendChild(errorModal);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (errorModal.parentNode) {
                errorModal.remove();
            }
        }, 10000);
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.chatInterface) {
        window.chatInterface.showError('An unexpected error occurred. Please refresh the page and try again.');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.chatInterface) {
        window.chatInterface.showError('An unexpected error occurred. Please refresh the page and try again.');
    }
});
