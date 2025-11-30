# 🤖 Autonomous QA Agent - RAG Enhanced

An intelligent Quality Assurance agent that dynamically generates test cases and Selenium scripts based on project documentation using Retrieval-Augmented Generation (RAG) and semantic search.

## 🚀 Features

### Core Capabilities
- **📁 Document Analysis**: Automatically extracts features, workflows, and UI elements from project documentation
- **🧪 Intelligent Test Generation**: Generates comprehensive test cases using RAG and semantic search
- **💻 Selenium Script Generation**: Creates executable Selenium WebDriver scripts from test cases
- **🔍 Semantic Search**: Finds relevant content across documents for context-aware test generation
- **🎯 Feature Detection**: Identifies key functionalities like login, checkout, search, forms, etc.

### Advanced Features
- **RAG-Enhanced Generation**: Uses document context for grounded, evidence-based test creation
- **Multi-Format Support**: Processes MD, TXT, JSON, HTML, and PDF files
- **Selector Mapping**: Automatically generates CSS/XPath selectors for UI elements
- **Quality Metrics**: Calculates confidence scores and evidence-based quality indicators
- **Batch Processing**: Handles multiple test cases and documents simultaneously


## 📋 Prerequisites

- Python 3.8 or higher
- Chrome Browser (for Selenium execution)
- ChromeDriver (automatically handled in scripts)

## 🛠️ Installation & Setup

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd autonomous-qa-agent/backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt


python -m venv venv
# Activate the environment (Windows)
.\venv\Scripts\activate
# Activate the environment (macOS/Linux)
# source venv/bin/activate

# Run from backend directory
python app.py

# Open new terminal and navigate to frontend directory
cd autonomous-qa-agent/frontend

# Install frontend dependencies
pip install streamlit requests

# Start Streamlit frontend
streamlit run app.py




