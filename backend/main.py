import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NexusIQ - Agentic Next Best Action Platform API",
    description="Backend API for multi-agent customer success decisioning.",
    version="0.1.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the path to frontend directory relative to main.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

import json
from retrieval import match_playbook

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "platform": "NexusIQ",
        "version": "0.1.0"
    }

@app.get("/api/customers")
async def get_customers():
    customers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "customers.json")
    if os.path.exists(customers_path):
        with open(customers_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

@app.get("/api/test-rag")
async def test_rag(query: str = ""):
    result = match_playbook(query)
    return {
        "Matched Playbook": result["playbook"],
        "Similarity Score": result["score"],
        "Matched Text": result["text"]
    }

@app.get("/api/signal/{customer}")
async def get_signal_analysis(customer: str):
    episodic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "episodic.json")
    raw_signal = ""
    
    if os.path.exists(episodic_path):
        with open(episodic_path, "r", encoding="utf-8") as f:
            try:
                interactions_data = json.load(f)
                for item in interactions_data:
                    if item.get("customer_id") == customer:
                        raw_signal = " ".join([i.get("content", "") for i in item.get("interactions", [])])
                        break
            except Exception as e:
                print(f"Error loading episodic memory: {e}")
                
    if not raw_signal:
        raw_signal = "No customer interactions found in episodic memory."
        
    from agents import signal_analyst
    analysis = signal_analyst(raw_signal)
    return {
        "customer_id": customer,
        "raw_signal": raw_signal,
        "analysis": analysis
    }

@app.get("/api/risk/{customer}")
async def get_risk_assessment(customer: str):
    # 1. Load customer metrics
    customers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "customers.json")
    metrics = None
    if os.path.exists(customers_path):
        with open(customers_path, "r", encoding="utf-8") as f:
            try:
                customers_data = json.load(f)
                for item in customers_data:
                    if item.get("id") == customer:
                        metrics = item
                        break
            except Exception as e:
                print(f"Error loading customers database: {e}")
                
    if not metrics:
        return {"error": f"Customer '{customer}' not found in database."}

    # 2. Load episodic memory (interactions)
    episodic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "episodic.json")
    raw_signal = ""
    if os.path.exists(episodic_path):
        with open(episodic_path, "r", encoding="utf-8") as f:
            try:
                interactions_data = json.load(f)
                for item in interactions_data:
                    if item.get("customer_id") == customer:
                        raw_signal = " ".join([i.get("content", "") for i in item.get("interactions", [])])
                        break
            except Exception as e:
                print(f"Error loading episodic memory: {e}")
                
    if not raw_signal:
        raw_signal = "No customer interactions found in episodic memory."

    # 3. Call Signal Analyst
    from agents import signal_analyst, risk_assessor
    signal_output = signal_analyst(raw_signal)

    # 4. Call Risk Assessor
    risk_output = risk_assessor(metrics, signal_output)

    # 5. Return chained result
    return {
        "customer_id": customer,
        "metrics": metrics,
        "signal_analysis": signal_output,
        "risk_assessment": risk_output
    }





@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
