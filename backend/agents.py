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


def context_builder(metrics: dict, signal_output: dict) -> dict:
    """
    Acts as the Devil's Advocate / Customer Advocate.
    Intentionally identifies positive opportunities, mitigating explanations, and expansion pathways.
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
You are the Context Builder (Advocate / Challenger) Agent in an Agentic Customer Success platform.
Your purpose is to look past immediate churn risks and identify hidden opportunities, positive signals, and alternative explanations.

Customer Profile Metrics:
{json.dumps(metrics, indent=2)}

Signal Analyst Findings:
{json.dumps(signal_output, indent=2)}

Determine:
1. Alternative Reasoning (A challenger perspective finding positive signs, explaining away risk flags, or showing why the customer might stay)
2. Growth Opportunities (A list of 2-3 specific expansion, training, upsell, or relationship-building opportunities)
3. Confidence (A float between 0.0 and 1.0)

Format the output strictly as a JSON object with these keys:
- alternative_reasoning (string)
- growth_opportunities (list of strings)
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
            print(f"[Context Builder] Anthropic API failed, trying fallbacks: {e}")

    # 2. Try Gemini if key is present
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
You are the Context Builder (Customer Advocate) Agent of NexusIQ. Identify positive opportunities and mitigating factors.

Customer Profile Metrics:
{json.dumps(metrics, indent=2)}

Signal Analyst Findings:
{json.dumps(signal_output, indent=2)}

Respond with a JSON object matching these exact keys:
- alternative_reasoning (Alternative positive perspective or risk mitigation reasoning)
- growth_opportunities (List of strings for upsell/relationship building)
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
            print(f"[Context Builder] Gemini API failed, trying fallbacks: {e}")

    # 3. Try OpenAI if key is present
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""
Provide a customer advocate analysis looking for growth and upsells.
Metrics: {json.dumps(metrics)}
Signals: {json.dumps(signal_output)}

Format output as a raw JSON object with keys: alternative_reasoning, growth_opportunities, confidence.
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
            print(f"[Context Builder] OpenAI API failed, trying fallbacks: {e}")

    # 4. Fallback to Heuristics Engine
    return run_context_heuristics(metrics, signal_output)


def run_context_heuristics(metrics: dict, signal_output: dict) -> dict:
    health = metrics.get("health_score", 100)
    topics = signal_output.get("topics", [])
    sentiment = signal_output.get("sentiment", "Neutral")
    
    opportunities = []
    
    # Heuristics for Innoflow Systems (Expansion)
    if "Account Expansion" in topics or health >= 80:
        alt_reasoning = f"Customer has an exceptional health index of {health}/100 and shows strong positive engagement. There is zero risk of churn. This account is primed for rapid expansion."
        opportunities = [
            "Finalize contract addendum for the pending 50 additional seat licenses.",
            "Propose multi-year contract options with a standard 10% volume discount.",
            "Introduce advanced collaboration modules to expand footprint into other departments."
        ]
        confidence = 0.95
        
    # Heuristics for CloudX Technologies (At Risk / Churn)
    elif "Competitor Threat" in topics and health < 50:
        alt_reasoning = "While the executive sponsor left and they are auditing software spend, the client has actively engaged us to discuss options. This is a critical window to establish a strong relationship with the incoming leadership team, demonstrate clear historical value, and reset their expectations."
        opportunities = [
            "Schedule a VIP welcome call with our Customer Success Director and the new incoming leadership.",
            "Conduct a comprehensive Value Utilization Audit to showcase how much efficiency they gained over the past year.",
            "Offer a customized enterprise bundle or flexible quarterly payment terms to address budget audit concerns."
        ]
        confidence = 0.82
        
    # Heuristics for Acme Corp / Apex Solutions (Warning / Integration Issues)
    elif "Technical Integration" in topics or "Performance Issues" in topics:
        alt_reasoning = "The customer's frustration stems entirely from initial onboarding blockers (CRM integration timeouts). Since key contacts are actively complaining and requesting support calls, engagement remains high. Resolving these technical hurdles immediately will build deep trust and clear the path for full product adoption."
        opportunities = [
            "Assign a dedicated solutions engineer for a live 1-on-1 CRM sync debugging call.",
            "Invite their admin team to our advanced developer integration workshop.",
            "Propose a post-onboarding optimization review to ensure custom workflows are optimized."
        ]
        confidence = 0.88
        
    # Heuristics for general/stable accounts (Vortex Media)
    else:
        alt_reasoning = f"Customer account is healthy and stable (Health: {health}/100). The current touchpoints are general or low priority, suggesting consistent usage with minimal friction. This is an ideal time to build advocacy."
        opportunities = [
            "Invite their CS lead to present a case study at our next user community event.",
            "Provide early access to the new beta feature pipeline to drive stickiness.",
            "Propose a routine check-in call to discuss long-term strategic roadmaps."
        ]
        confidence = 0.90
        
    return {
        "alternative_reasoning": alt_reasoning,
        "growth_opportunities": opportunities,
        "confidence": confidence
    }


def synthesizer(signal_output: dict, risk_output: dict, context_output: dict, playbook_text: str) -> dict:
    """
    Acts as a Senior Manager / Synthesizer.
    Consolidates the findings from the Signal Analyst, Risk Assessor, and Context Builder,
    incorporates retrieved organizational playbook recommendations, and delivers final consensus actions.
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
You are the Decision Synthesizer Agent in an Agentic Next Best Action Platform.
Your role is like a Senior Customer Success Director who reviews all reports, cross-references corporate playbooks,
and makes final, structured recommendations.

Signal Analyst Report:
{json.dumps(signal_output, indent=2)}

Risk Assessor Report:
{json.dumps(risk_output, indent=2)}

Context Builder Report:
{json.dumps(context_output, indent=2)}

Retrieved Playbook Content:
\"\"\"
{playbook_text}
\"\"\"

Produce the final decision synthesis. Format the output strictly as a JSON object with these keys:
- consensus (string: summarized analysis of the conflicting reports and playbook alignment)
- recommendations (list of 3 strings: the top 3 actionable recommendations)
- confidence (float between 0.0 and 1.0)
- if_acted (string: expected business outcome if acted upon)
- if_ignored (string: business risk if ignored)

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
            print(f"[Synthesizer] Anthropic API failed, trying fallbacks: {e}")

    # 2. Try Gemini if key is present
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
You are the Decision Synthesizer Agent of NexusIQ. Consolidate analysis and playbooks into a final next-best-action synthesis.

Signal Findings: {json.dumps(signal_output)}
Risk Findings: {json.dumps(risk_output)}
Context Findings: {json.dumps(context_output)}
Playbook Guidelines: {playbook_text}

Respond with a JSON object matching these exact keys:
- consensus (Summary alignment of reports)
- recommendations (List of 3 actionable items)
- confidence (Float between 0.0 and 1.0)
- if_acted (Business outcome if acted)
- if_ignored (Business risk if ignored)

Respond ONLY with raw JSON. No markdown wrappers.
"""
            response = model.generate_content(prompt)
            text_res = response.text.strip()
            if text_res.startswith("```"):
                text_res = re.sub(r"^```(?:json)?\n", "", text_res)
                text_res = re.sub(r"\n```$", "", text_res)
            return json.loads(text_res.strip())
        except Exception as e:
            print(f"[Synthesizer] Gemini API failed, trying fallbacks: {e}")

    # 3. Try OpenAI if key is present
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""
Synthesize CS analysis using:
Signals: {json.dumps(signal_output)}
Risk: {json.dumps(risk_output)}
Context: {json.dumps(context_output)}
Playbook: {playbook_text}

Format output as a raw JSON object with keys: consensus, recommendations, confidence, if_acted, if_ignored.
recommendations must be a list of 3 strings.
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
            print(f"[Synthesizer] OpenAI API failed, trying fallbacks: {e}")

    # 4. Fallback to Heuristics Engine
    return run_synthesizer_heuristics(signal_output, risk_output, context_output, playbook_text)


def run_synthesizer_heuristics(signal_output: dict, risk_output: dict, context_output: dict, playbook_text: str) -> dict:
    topics = signal_output.get("topics", [])
    risk_level = risk_output.get("risk_level", "Medium")
    
    # 1. Expansion Case
    if "Account Expansion" in topics or risk_level == "Low":
        consensus = "The account is in an excellent healthy state. There is active buyer intent to purchase 50 additional seat licenses. We will align with their expansion timeline to maximize upsell value while maintaining system health."
        recommends = [
            "Proactively finalize and send over the contract addendum for the 50 additional seat licenses.",
            "Schedule a post-onboarding training workshop for the new CSM users joining next week.",
            "Schedule a multi-year contract options review with their CS Director to offer long-term loyalty pricing."
        ]
        confidence = 0.95
        if_acted = "Will secure immediate upsell expansion revenue, increase seats by 50, and establish a long-term contract lock-in."
        if_ignored = "Could delay the onboarding of their team members, causing integration adoption friction, and miss out on long-term renewal options."
        
    # 2. Competitor Threat Churn Case
    elif "Competitor Threat" in topics:
        consensus = "The account is at critical risk due to competitor evaluation and executive sponsor departure. However, the client's willingness to review options provides a vital window for intervention. We must align with our Competitor Threat playbook to establish new executive sponsors and prove value."
        recommends = [
            "Schedule an urgent alignment call with the new incoming VP of Operations and our CS Director.",
            "Conduct a comprehensive Value Utilization Audit showcasing platform ROI over the last 12 months.",
            "Offer a customized enterprise bundle or flexible quarterly payment terms to address budget audit concerns."
        ]
        confidence = 0.86
        if_acted = "Will de-escalate competitor threat, establish relationship with new leadership, and salvage $320,000 ARR contract."
        if_ignored = "High probability of complete churn within 60-90 days, resulting in a loss of $320,000 ARR."
        
    # 3. Technical Integration Warning Case
    elif "Technical Integration" in topics or "Performance Issues" in topics:
        consensus = "The account is in a Warning state due to initial data sync timeouts. Since key contacts are actively complaining and requesting support, they are highly engaged. Resolving the CRM synchronization bug immediately is the primary path to success."
        recommends = [
            "Assign a senior integrations engineer to debug the CRM sync timeout issues in a live 1-on-1 call today.",
            "Establish automated daily sync health alerts for the customer's operations team to rebuild confidence.",
            "Offer a post-sync workflow training session for the team to accelerate product utilization."
        ]
        confidence = 0.90
        if_acted = "Will resolve the sync blockers, drive daily active usage, and restore the account status back to Healthy."
        if_ignored = "Sync timeout will lead to implementation failure, client requesting a refund, and high probability of churn at renewal."
        
    # 4. General / Stable case
    else:
        consensus = "The account is stable with steady usage and low support ticket activity. No active risks or critical opportunities exist. This is the optimal window to convert the account into a reference advocate."
        recommends = [
            "Invite their Customer Success lead to present a case study at our next user community event.",
            "Grant early access to the upcoming beta feature pipeline to drive platform stickiness.",
            "Schedule a standard business check-in call to align on their long-term feature roadmap request."
        ]
        confidence = 0.88
        if_acted = "Will convert a stable account into an active brand advocate, generating customer marketing collateral."
        if_ignored = "Missed opportunity to build brand advocacy; the account remains stable but static."
        
    return {
        "consensus": consensus,
        "recommendations": recommends,
        "confidence": confidence,
        "if_acted": if_acted,
        "if_ignored": if_ignored
    }


