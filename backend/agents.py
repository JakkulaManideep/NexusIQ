import os
import json
import re

def signal_analyst(raw_signal: str) -> dict:
    """
    Extracts structured metrics (Sentiment, Topics, Pain Points, Urgency) from a raw customer interaction log.
    Utilizes real LLMs if API keys are set, otherwise falls back to a deterministic heuristics engine.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    # 1. Try Anthropic (Claude) if key is present
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            prompt = f"""
You are the Signal Analyst Agent in an Agentic Next Best Action Platform.
Analyze the raw customer interaction signal below and extract structural signals.

Input Raw Signal:
\"\"\"
{raw_signal}
\"\"\"

Format the output strictly as a JSON object with these keys:
- sentiment (string: 'Positive', 'Neutral', 'Warning', 'Negative', 'Mixed')
- topics (list of strings, e.g., 'Competitor Threat', 'Technical Integration', 'Billing', etc.)
- pain_points (list of strings describing specific challenges mentioned)
- urgency (string: 'Low', 'Medium', 'High', 'Critical')

Return ONLY the raw JSON string. Do not include markdown codeblocks or text before/after.
"""
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system="You are a strict JSON generator. Never return prose.",
                messages=[{"role": "user", "content": prompt}]
            )
            text_res = message.content[0].text.strip()
            if text_res.startswith("```"):
                text_res = re.sub(r"^```(?:json)?\n", "", text_res)
                text_res = re.sub(r"\n```$", "", text_res)
            return json.loads(text_res.strip())
        except Exception as e:
            print(f"[Signal Analyst] Anthropic API failed, trying fallbacks: {e}")

    # 2. Try Gemini if key is present
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
You are the Signal Analyst Agent of NexusIQ. Analyze this customer raw signal. Extract:
- Sentiment: One of ['Positive', 'Neutral', 'Warning', 'Negative', 'Mixed']
- Topics: List of strings (e.g., 'Competitor Threat', 'Technical Integration', 'Billing', etc.)
- Pain Points: List of strings detailing the specific problems mentioned by the customer.
- Urgency: One of ['Low', 'Medium', 'High', 'Critical']

Input Raw Signal:
{raw_signal}

Respond ONLY with a valid JSON object matching the keys above. Do not wrap it in markdown code blocks.
"""
            response = model.generate_content(prompt)
            text_res = response.text.strip()
            if text_res.startswith("```"):
                text_res = re.sub(r"^```(?:json)?\n", "", text_res)
                text_res = re.sub(r"\n```$", "", text_res)
            return json.loads(text_res.strip())
        except Exception as e:
            print(f"[Signal Analyst] Gemini API failed, trying fallbacks: {e}")

    # 3. Try OpenAI if key is present
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""
Analyze the customer interaction raw signal and extract:
- sentiment (Positive, Neutral, Warning, Negative, Mixed)
- topics (List of strings)
- pain_points (List of strings)
- urgency (Low, Medium, High, Critical)

Input: {raw_signal}

Format output as a raw JSON object.
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            text_res = response.choices[0].message.content.strip()
            return json.loads(text_res)
        except Exception as e:
            print(f"[Signal Analyst] OpenAI API failed, trying fallbacks: {e}")

    # 4. Fallback to Local Heuristics Engine
    return run_heuristics_analysis(raw_signal)

def run_heuristics_analysis(raw_signal: str) -> dict:
    text = raw_signal.lower()
    
    # 1. Sentiment Heuristics
    sentiment = "Neutral"
    pos_words = ["love", "happy", "save", "helpful", "great", "thanks", "perfect", "appreciate", "loving", "upgraded"]
    neg_words = ["slow", "fail", "error", "issue", "struggle", "degradation", "timeout", "terminate", "refund", "downgrade", "complained", "pricing", "limit"]
    
    pos_count = sum(1 for w in pos_words if w in text)
    neg_count = sum(1 for w in neg_words if w in text)
    
    if pos_count > 0 and neg_count > 0:
        sentiment = "Mixed"
    elif pos_count > 1 and neg_count == 0:
        sentiment = "Positive"
    elif neg_count > 1:
        sentiment = "Negative"
    elif neg_count == 1:
        sentiment = "Warning"
        
    # 2. Topics Heuristics
    topics = []
    if any(w in text for w in ["gong", "zoominfo", "clari", "competitor", "alternative"]):
        topics.append("Competitor Threat")
    if any(w in text for w in ["sync", "crm", "api", "integration", "timeout", "error"]):
        topics.append("Technical Integration")
    if any(w in text for w in ["price", "pricing", "budget", "cost", "spend", "refund"]):
        topics.append("Pricing & Budget")
    if any(w in text for w in ["upgrade", "license", "seat", "purchase", "onboarding"]):
        topics.append("Account Expansion")
    if any(w in text for w in ["slow", "degradation", "performance", "speed"]):
        topics.append("Performance Issues")
    if not topics:
        topics.append("General Inquiries")
        
    # 3. Pain Points Heuristics
    pain_points = []
    if "performance degradation" in text or "slow" in text:
        pain_points.append("Product performance and interface speed degradation")
    if "crm sync" in text or "crm data sync" in text or "sync is failing" in text:
        pain_points.append("CRM data synchronization failures and timeouts")
    if "gong" in text or "competitor" in text or "zoominfo" in text:
        pain_points.append("Customer evaluating competitors (Gong/ZoomInfo)")
    if "pricing" in text or "refund" in text or "spend" in text:
        pain_points.append("Pricing complaints and audit of software budget")
    if "timeout" in text or "error" in text:
        pain_points.append("API error rates and connection issues")
    
    if not pain_points:
        if neg_count > 0:
            pain_points.append("General usage friction points")
        else:
            pain_points.append("No active pain points detected")
            
    # 4. Urgency Heuristics
    urgency = "Low"
    if any(w in text for w in ["urgent", "blocker", "critical", "refund", "terminate"]):
        urgency = "Critical"
    elif any(w in text for w in ["tomorrow", "next week", "evaluating next month", "deadline"]):
        urgency = "High"
    elif any(w in text for w in ["check", "question", "no rush"]):
        urgency = "Low"
    elif neg_count > 0:
        urgency = "Medium"
        
    return {
        "sentiment": sentiment,
        "topics": topics,
        "pain_points": pain_points,
        "urgency": urgency
    }
