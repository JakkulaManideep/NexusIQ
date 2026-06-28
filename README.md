# NexusIQ – Agentic Next Best Action Platform

NexusIQ is an **Agentic Decision Intelligence Platform** designed for Customer Success Managers (CSMs) in B2B SaaS companies. Instead of a standard chatbot or simple retrieval-augmented generation (RAG) tool, NexusIQ orchestrates a team of specialized AI agents working together to analyze customer interactions, weigh risks, explore growth opportunities, and synthesize explainable next-best-action recommendations.

---

## 🚀 Key Features

* **Multi-Agent Orchestration**: Sequential cognitive chain featuring:
  1. **Signal Analyst Agent**: Extracts sentiment, topics, pain points, and urgency from raw timeline signals.
  2. **Risk Assessor Agent**: Audits telemetry and signals to flag churn risk and calculate confidence.
  3. **Context Builder Agent**: Acts as the Devil's Advocate to identify expansion/upsell opportunities and alternative explanations.
  4. **Decision Synthesizer Agent**: Merges all analysis with standard operating playbooks to generate final consensus recommendations.
* **Explainable AI (RAG Playbook Match)**: Uses a local TF-IDF semantic vector space to dynamically retrieve and match custom CSM playbooks based on user telemetry.
* **Three-Layer Memory Architecture**:
  * **Episodic Memory**: Raw timeline interactions, support logs, and emails (`episodic.json`).
  * **Semantic Memory**: Corporate playbooks and guidelines (`playbooks/`).
  * **Strategic Memory**: Persistent feedback loop storing accepted, modified, or rejected recommendations (`strategic.json`).
* **Interactive CS Cockpit**: A premium dark-theme dashboard designed with a Plus Jakarta Sans typeface, circular health gauges, sequential card cascade animations, custom toast notifications, and an interactive architecture inspector modal.
* **Robust Hybrid Engine**: Seamlessly integrates with **Anthropic Claude**, **Gemini**, or **OpenAI** APIs when keys are available in environment variables, and falls back to a deterministic local heuristics engine otherwise.

---

## 🛠️ Technology Stack

* **Backend**: FastAPI, Python 3.12, Uvicorn, Scikit-Learn (TF-IDF Vectorization)
* **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (via CDN)
* **Storage**: Local JSON databases (file-based persistence)

---

## 📂 Project Structure

```
NexusIQ/
├── backend/
│   ├── data/
│   │   ├── playbooks/             # Standard operating playbook text files
│   │   ├── customers.json         # Customer accounts metadata database
│   │   ├── episodic.json          # Episodic customer interaction logs
│   │   └── strategic.json         # Persistent strategic feedback memory
│   ├── agents.py                  # AI Agent definition chains & heuristics fallback
│   ├── main.py                    # FastAPI entrypoint, API routes & index delivery
│   ├── memory.py                  # Read/write interfaces for Strategic Memory
│   ├── requirements.txt           # Python backend dependencies
│   └── retrieval.py               # TF-IDF RAG Document vectorizer & matching
├── frontend/
│   └── index.html                 # Main single-page CS Cockpit dashboard
└── README.md                      # Project documentation
```

---

## 🏃 How to Run the Demo

### 1. Configure the Virtual Environment
Navigate to the root directory and create/activate the Python virtual environment:
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

### 3. (Optional) Configure LLM Keys
Set any of the following environment variables to activate live LLM orchestration:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
$env:OPENAI_API_KEY="your-openai-key"
```
*Note: If no API keys are provided, the platform automatically switches to a deterministic heuristics engine modeled after the customer data profiles, ensuring a fully functional demo experience.*

### 4. Start the Server
Run the FastAPI application:
```powershell
cd backend
python main.py
```
Open your browser and navigate to **`http://127.0.0.1:8000`** to access the Cockpit.

---

## 👥 Multi-Agent Consensus Workflow

```
Customer Selection ➔ Run Analysis
        ↓
[1. Signal Analyst] ➔ Sentiment, Urgency, Pain Points
        ↓
┌───────────────────────┴───────────────────────┐
↓                                               ↓
[2. Risk Assessor]                      [3. Context Builder]
Churn Threats, Confidence               Advocate viewpoints, Upsells
└───────────────────────┬───────────────────────┘
                        ↓
             [playbooks/ RAG Retrieval]
                        ↓
            [4. Decision Synthesizer] ➔ Consensus Summary
                        ↓
             Ranked recommendations (Accept/Modify/Reject)
                        ↓
             Strategic Memory Logging
```

---

## 🏆 Hackathon Goals Met
* **Human-in-the-loop**: Interactive CSM feedback actions directly stored to Strategic Memory.
* **Aesthetics & UX**: Cascading slide-up fade animations, circular health charts, and custom toast alerts.
* **Modularity**: Clean separation of backend APIs and the frontend HTML.
