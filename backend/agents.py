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


def risk_assessor(metrics: dict, signal_output: dict) -> dict:
    """
    Evaluates customer churn risk, service threats, and dissatisfaction.
    Takes the Customer Profile (metrics) and the Signal Analyst's findings (signal_output).
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
You are the Risk Assessor Agent in an Agentic Next Best Action Platform.
Evaluate the overall churn and business risk for this customer.

Customer Profile Metrics:
{json.dumps(metrics, indent=2)}

Signal Analyst Findings:
{json.dumps(signal_output, indent=2)}

Determine:
1. Risk Level (One of 'Low', 'Medium', 'High', 'Critical')
2. Reason (A robust business-oriented explanation factoring in both the telemetry metrics and user signals)
3. Confidence (A float between 0.0 and 1.0)

Format the output strictly as a JSON object with these keys:
- risk_level (string)
- reason (string)
- confidence (float)

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
            print(f"[Risk Assessor] Anthropic API failed, trying fallbacks: {e}")

    # 2. Try Gemini if key is present
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
You are the Risk Assessor Agent of NexusIQ. Evaluate churn and business risk.

Customer Profile Metrics:
{json.dumps(metrics, indent=2)}

Signal Analyst Findings:
{json.dumps(signal_output, indent=2)}

Respond with a JSON object matching these exact keys:
- risk_level (One of ['Low', 'Medium', 'High', 'Critical'])
- reason (Detailed professional explanation)
- confidence (Float between 0.0 and 1.0)

Respond ONLY with raw JSON. No markdown wrappers.
"""
            response = model.generate_content(prompt)
            text_res = response.text.strip()
            if text_res.startswith("```"):
                text_res = re.sub(r"^```(?:json)?\n", "", text_res)
                text_res = re.sub(r"\n```$", "", text_res)
            return json.loads(text_res.strip())
        except Exception as e:
            print(f"[Risk Assessor] Gemini API failed, trying fallbacks: {e}")

    # 3. Try OpenAI if key is present
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""
Evaluate customer risk using:
Metrics: {json.dumps(metrics)}
Signals: {json.dumps(signal_output)}

Format output as a raw JSON object with keys: risk_level, reason, confidence.
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
            print(f"[Risk Assessor] OpenAI API failed, trying fallbacks: {e}")

    # 4. Fallback to Local Risk Heuristics Engine
    return run_risk_heuristics(metrics, signal_output)


def run_risk_heuristics(metrics: dict, signal_output: dict) -> dict:
    health = metrics.get("health_score", 100)
    arr = metrics.get("arr", 0)
    status = metrics.get("status", "Stable")
    
    sentiment = signal_output.get("sentiment", "Neutral")
    topics = signal_output.get("topics", [])
    pain_points = signal_output.get("pain_points", [])
    urgency = signal_output.get("urgency", "Low")
    
    # Calculate a numerical risk score (0 to 100)
    # 1. Base score from health index (inverse relationship)
    risk_score = 100 - health
    
    # 2. Adjust based on raw signal findings
    if sentiment == "Negative":
        risk_score += 20
    elif sentiment == "Warning" or sentiment == "Mixed":
        risk_score += 10
        
    if "Competitor Threat" in topics:
        risk_score += 25
    if "Technical Integration" in topics:
        risk_score += 15
    if "Performance Issues" in topics:
        risk_score += 10
    if "Pricing & Budget" in topics:
        risk_score += 15
        
    if urgency == "Critical":
        risk_score += 20
    elif urgency == "High":
        risk_score += 10
        
    # Cap between 0 and 100
    risk_score = max(0, min(100, risk_score))
    
    # Determine Risk Level
    if risk_score >= 70 or status == "At Risk":
        risk_level = "Critical"
    elif risk_score >= 50 or status == "Warning":
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # Build detailed reasoning string
    reasons = []
    if health < 70:
        reasons.append(f"customer health index is low at {health}/100")
    if "Competitor Threat" in topics:
        reasons.append("customer is actively evaluating competitors")
    if "Technical Integration" in topics:
        reasons.append("unresolved CRM/API integration failure is blocking setup")
    if "Performance Issues" in topics:
        reasons.append("system latency and performance degradation complaints")
    if "Pricing & Budget" in topics:
        reasons.append("pricing complaints and threats of requesting a refund/spend review")
    if urgency in ["High", "Critical"] and neg_words_found_in_points(pain_points):
        reasons.append(f"customer flagged high-urgency pain points ({', '.join(pain_points[:2])})")
        
    if not reasons:
        reason_str = f"The customer's status is stable. Health score is {health}/100 with standard usage patterns and no active critical pain points detected."
    else:
        # Join reasons with commas
        reasons_joined = ", ".join(reasons)
        # Capitalize first letter
        reason_str = f"Risk evaluated as {risk_level} because: {reasons_joined}. Immediate CSM outreach is recommended to mitigate churn threat."
        
    # Determine confidence score
    confidence = 0.85
    if "Competitor Threat" in topics and health < 60:
        confidence = 0.92  # High correlation makes us more confident
    elif sentiment == "Neutral" and health > 80:
        confidence = 0.88
    elif sentiment == "Negative" and health > 80:
        confidence = 0.78  # Conflicting data decreases confidence slightly
        
    return {
        "risk_level": risk_level,
        "reason": reason_str,
        "confidence": confidence
    }

def neg_words_found_in_points(pain_points):
    return len(pain_points) > 0 and pain_points[0] != "No active pain points detected"
