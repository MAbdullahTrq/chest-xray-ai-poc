// Frontend JavaScript for Chest X-ray AI Diagnostic Tool

class ChestXRayApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentFile = null;
        this.processingStartTime = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkApiStatus();
        
        // Update processing time during analysis
        this.processingTimer = null;
    }

    initializeElements() {
        // Main elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImage = document.getElementById('previewImage');
        this.imageInfo = document.getElementById('imageInfo');
        this.removeImageBtn = document.getElementById('removeImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        
        // Status elements
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.processingStatus = document.getElementById('processingStatus');
        this.resultsContent = document.getElementById('resultsContent');
        this.processingTime = document.getElementById('processingTime');
        this.finalProcessingTime = document.getElementById('finalProcessingTime');
        this.analysisDate = document.getElementById('analysisDate');
        this.topPrediction = document.getElementById('topPrediction');
        this.findingsList = document.getElementById('findingsList');
        this.recommendationsContent = document.getElementById('recommendationsContent');
        
        // Error elements
        this.errorSection = document.getElementById('errorSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');
        
        // Action buttons
        this.downloadReportBtn = document.getElementById('downloadReport');
        this.analyzeAnotherBtn = document.getElementById('analyzeAnother');
        
        // Modal elements
        this.modal = document.getElementById('modal');
        this.modalClose = document.getElementById('modalClose');
        this.modalBody = document.getElementById('modalBody');
        
        // Footer links
        this.aboutLink = document.getElementById('aboutLink');
        this.apiDocsLink = document.getElementById('apiDocsLink');
        this.helpLink = document.getElementById('helpLink');
    }

    attachEventListeners() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Buttons
        this.removeImageBtn.addEventListener('click', this.removeImage.bind(this));
        this.analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));
        this.retryBtn.addEventListener('click', this.retryAnalysis.bind(this));
        this.downloadReportBtn.addEventListener('click', this.downloadReport.bind(this));
        this.analyzeAnotherBtn.addEventListener('click', this.analyzeAnother.bind(this));
        
        // Modal events
        this.modalClose.addEventListener('click', this.closeModal.bind(this));
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) this.closeModal();
        });
        
        // Footer links
        this.aboutLink.addEventListener('click', (e) => {
            e.preventDefault();
            this.showAbout();
        });
        this.apiDocsLink.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(`${this.apiBaseUrl}/docs`, '_blank');
        });
        this.helpLink.addEventListener('click', (e) => {
            e.preventDefault();
            this.showHelp();
        });
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (response.ok && data.status === 'healthy') {
                this.updateStatus('online', 'API Connected');
            } else {
                this.updateStatus('offline', 'API Unavailable');
            }
        } catch (error) {
            this.updateStatus('offline', 'API Offline');
            console.error('API status check failed:', error);
        }
    }

    updateStatus(status, message) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = message;
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom'];
        if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm')) {
            this.showError('Unsupported file type. Please upload JPEG, PNG, or DICOM files.');
            return;
        }

        // Validate file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showError(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 10MB.`);
            return;
        }

        this.currentFile = file;
        this.displayImagePreview(file);
        this.analyzeBtn.disabled = false;
        this.hideError();
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.imageInfo.innerHTML = `
                <strong>${file.name}</strong><br>
                Size: ${(file.size / 1024).toFixed(1)} KB<br>
                Type: ${file.type || 'DICOM'}
            `;
            this.imagePreview.style.display = 'flex';
            this.uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.currentFile = null;
        this.imagePreview.style.display = 'none';
        this.uploadArea.style.display = 'block';
        this.analyzeBtn.disabled = true;
        this.fileInput.value = '';
        this.hideResults();
        this.hideError();
    }

    async analyzeImage() {
        if (!this.currentFile) return;

        this.showProcessingStatus();
        this.processingStartTime = Date.now();
        
        // Start processing timer
        this.processingTimer = setInterval(() => {
            if (this.processingStartTime) {
                const elapsed = (Date.now() - this.processingStartTime) / 1000;
                this.processingTime.textContent = elapsed.toFixed(1);
            }
        }, 100);

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.displayResults(data);
            } else {
                throw new Error(data.message || 'Analysis failed');
            }
        } catch (error) {
            this.showError(`Analysis failed: ${error.message}`);
            console.error('Analysis error:', error);
        } finally {
            this.hideProcessingStatus();
            if (this.processingTimer) {
                clearInterval(this.processingTimer);
                this.processingTimer = null;
            }
        }
    }

    showProcessingStatus() {
        this.resultsSection.style.display = 'block';
        this.processingStatus.style.display = 'block';
        this.resultsContent.style.display = 'none';
        this.analyzeBtn.disabled = true;
        this.hideError();
    }

    hideProcessingStatus() {
        this.processingStatus.style.display = 'none';
        this.analyzeBtn.disabled = false;
    }

    displayResults(data) {
        // Update processing time and date
        this.finalProcessingTime.textContent = `${data.processing_time}s`;
        this.analysisDate.textContent = new Date().toLocaleString();

        // Display top prediction
        const topPredictionElement = this.topPrediction;
        const predictionName = topPredictionElement.querySelector('.prediction-name');
        const confidenceScore = topPredictionElement.querySelector('.confidence-score');
        
        predictionName.textContent = data.top_prediction;
        confidenceScore.textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Apply color coding based on confidence
        if (data.confidence > 0.7) {
            confidenceScore.style.color = 'var(--error-color)';
        } else if (data.confidence > 0.5) {
            confidenceScore.style.color = 'var(--warning-color)';
        } else {
            confidenceScore.style.color = 'var(--success-color)';
        }

        // Display all findings
        this.displayFindings(data.findings);

        // Display recommendations
        this.recommendationsContent.innerHTML = `<p>${data.recommendations}</p>`;

        // Show results
        this.resultsContent.style.display = 'block';
        this.hideError();
    }

    displayFindings(findings) {
        // Sort findings by confidence (descending)
        const sortedFindings = Object.entries(findings)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8); // Show top 8 findings

        this.findingsList.innerHTML = sortedFindings.map(([name, confidence]) => {
            const confidenceClass = confidence > 0.7 ? 'high-confidence' : 
                                  confidence > 0.4 ? 'medium-confidence' : 'low-confidence';
            
            return `
                <div class="finding-item ${confidenceClass}">
                    <span class="finding-name">${name}</span>
                    <div class="finding-confidence">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                        </div>
                        <span class="confidence-value">${(confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.style.display = 'block';
        this.hideResults();
    }

    hideError() {
        this.errorSection.style.display = 'none';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }

    retryAnalysis() {
        this.hideError();
        if (this.currentFile) {
            this.analyzeImage();
        }
    }

    downloadReport() {
        // Create a simple text report
        const findings = Array.from(this.findingsList.children).map(item => {
            const name = item.querySelector('.finding-name').textContent;
            const confidence = item.querySelector('.confidence-value').textContent;
            return `${name}: ${confidence}`;
        }).join('\n');

        const report = `
CHEST X-RAY AI ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}
File: ${this.currentFile?.name || 'Unknown'}

PRIMARY FINDING:
${this.topPrediction.querySelector('.prediction-name').textContent}: ${this.topPrediction.querySelector('.confidence-score').textContent}

ALL FINDINGS:
${findings}

CLINICAL RECOMMENDATIONS:
${this.recommendationsContent.textContent}

PROCESSING TIME: ${this.finalProcessingTime.textContent}

---
This report was generated by an AI system and should be reviewed by a qualified healthcare professional.
        `.trim();

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chest_xray_analysis_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    analyzeAnother() {
        this.removeImage();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    showAbout() {
        this.modalBody.innerHTML = `
            <h2>About Chest X-ray AI Diagnostic Tool</h2>
            <p>This is a Proof of Concept (POC) application that uses artificial intelligence to analyze chest X-ray images for potential pathologies.</p>
            
            <h3>Features:</h3>
            <ul>
                <li>Automatic detection of 14+ chest pathologies</li>
                <li>Real-time analysis with confidence scores</li>
                <li>Support for JPEG, PNG, and DICOM formats</li>
                <li>Clinical recommendations based on findings</li>
                <li>Downloadable analysis reports</li>
            </ul>
            
            <h3>Technology:</h3>
            <ul>
                <li>TorchXRayVision pretrained models</li>
                <li>FastAPI backend</li>
                <li>Modern web interface</li>
            </ul>
            
            <p><strong>Important:</strong> This tool is for educational and research purposes only. All results should be reviewed by qualified healthcare professionals.</p>
        `;
        this.modal.style.display = 'block';
    }

    showHelp() {
        this.modalBody.innerHTML = `
            <h2>How to Use</h2>
            
            <h3>1. Upload X-ray Image</h3>
            <p>Click the upload area or drag and drop your chest X-ray image. Supported formats:</p>
            <ul>
                <li>JPEG/JPG files</li>
                <li>PNG files</li>
                <li>DICOM files (.dcm)</li>
            </ul>
            <p>Maximum file size: 10MB</p>
            
            <h3>2. Analyze Image</h3>
            <p>Click the "Analyze X-ray" button to start the AI analysis. Processing typically takes 2-5 seconds.</p>
            
            <h3>3. Review Results</h3>
            <p>The system will display:</p>
            <ul>
                <li>Primary finding with confidence score</li>
                <li>All detected pathologies with probabilities</li>
                <li>Clinical recommendations</li>
                <li>Processing time and analysis date</li>
            </ul>
            
            <h3>4. Download Report</h3>
            <p>You can download a text report of the analysis for your records.</p>
            
            <h3>Supported Pathologies:</h3>
            <p>Pneumonia, Pneumothorax, Pleural Effusion, Atelectasis, Cardiomegaly, Consolidation, Edema, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening</p>
            
            <p><strong>Disclaimer:</strong> This is an AI-assisted tool for educational purposes. Always consult with healthcare professionals for medical diagnosis.</p>
        `;
        this.modal.style.display = 'block';
    }

    closeModal() {
        this.modal.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChestXRayApp();
});

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const modal = document.getElementById('modal');
        if (modal.style.display === 'block') {
            modal.style.display = 'none';
        }
    }
});
