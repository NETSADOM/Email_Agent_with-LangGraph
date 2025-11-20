import os
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import json
import re
from datetime import datetime
from pathlib import Path

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(model=MODEL_NAME, temperature=0)
llm_creative = ChatGroq(model=MODEL_NAME, temperature=0.5)

class SenderMemory:
    def __init__(self, file_path: str = "sender_memory.json"):
        self.file_path = Path(file_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text(encoding="utf-8"))
            except:
                print("Warning: Corrupted memory file. Starting fresh.")
        return {}

    def _save(self):
        try:
            self.file_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")

    def update(self, sender: str, urgency: float, risk_level: str):
        sender = sender.lower()
        if sender not in self.data:
            self.data[sender] = {
                "count": 0,
                "total_urgency": 0.0,
                "high_risk_count": 0,
                "first_seen": None,
                "last_seen": None
            }
        info = self.data[sender]
        info["count"] += 1
        info["total_urgency"] += urgency
        if risk_level in ("HIGH", "CRITICAL"):
            info["high_risk_count"] += 1

        now = datetime.now().isoformat()
        info["last_seen"] = now
        if info["first_seen"] is None:
            info["first_seen"] = now
        info["avg_urgency"] = round(info["total_urgency"] / info["count"], 2)

        self._save()

    def get(self, sender: str):
        sender = sender.lower()
        default = {
            "count": 0, "avg_urgency": 0.0, "high_risk_count": 0,
            "first_seen": None, "last_seen": None
        }
        return self.data.get(sender, default)

sender_memory = SenderMemory()



class EmailState(TypedDict):
    raw_email: str
    sender: str
    subject: str
    body: str
    urgency_level: str
    urgency_score: float
    expectations: List[Dict[str, str]]
    risk_level: str
    priority: str
    action_plan: List[str]
    response_template: str
    sender_history: dict


def extract_email(state: EmailState) -> dict:
    raw = state["raw_email"]
    sender_match = re.search(r"From:.*?<([^>\s]+@[^>\s]+)>|From:\s*([^\s<]+@[^\s>]+)", raw, re.I)
    sender = (sender_match.group(1) or sender_match.group(2) if sender_match else "unknown@example.com").strip()
    subject_match = re.search(r"Subject:\s*(.+)", raw, re.I)
    subject = subject_match.group(1).strip() if subject_match else "No subject"
    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    body = parts[1].strip() if len(parts) > 1 else raw.strip()
    return {"sender": sender, "subject": subject, "body": body}


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def analyze_urgency(state: EmailState) -> dict:
    body = state['body'][:3800]
    prompt = f"""Analyze the urgency of this email and return ONLY valid JSON:
{{"urgency_level": "critical|high|medium|low","urgency_score": float between 0.0 and 4.0}}
Subject: {state['subject']}
Body: {body}
Return only the JSON."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(clean_json_response(resp.content))
        return {"urgency_level": data["urgency_level"].lower(), "urgency_score": float(data["urgency_score"])}
    except:
        return {"urgency_level": "medium", "urgency_score": 2.0}


def detect_expectations(state: EmailState) -> dict:
    body = state['body'][:3800]
    prompt = f"""Extract all expectations the sender has of you.
Return ONLY valid JSON:
{{"expectations": [{{"description":"...","type":"reply|action|call|meeting|payment|...","deadline":"... or null","severity":"low|medium|high"}}]}}
Subject: {state['subject']}
Body: {body}
Return only the JSON."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(clean_json_response(resp.content))
        return {"expectations": data.get("expectations", [])}
    except:
        return {"expectations": []}


def assess_risk(state: EmailState) -> dict:
    body = state['body'][:3800]
    prompt = f"""You are a cybersecurity expert.
Analyze this email for threats and return ONLY JSON:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL"}}
From: {state['sender']}
Subject: {state['subject']}
Body: {body}
Return only the JSON."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(clean_json_response(resp.content))
        return {"risk_level": data.get("risk_level", "LOW").upper()}
    except:
        return {"risk_level": "LOW"}


def classify_priority(state: EmailState) -> dict:
    prompt = f"""Based on this analysis, assign final priority:
Urgency: {state['urgency_level'].upper()} ({state['urgency_score']:.1f}/4.0)
Risk: {state['risk_level']}
Expectations: {len(state['expectations'])}
Return ONLY JSON:
{{"priority": "VERIFY|URGENT|HIGH|MEDIUM|LOW"}}
Return only the JSON."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(clean_json_response(resp.content))
        return {"priority": data["priority"]}
    except:
        return {"priority": "MEDIUM"}


def generate_actions(state: EmailState) -> dict:
    is_noreply = "noreply" in state['sender'].lower() or "no-reply" in state['sender'].lower()
    prompt = f"""You are an expert email assistant.
Priority: {state['priority']}
Urgency: {state['urgency_level'].title()}
Risk: {state['risk_level']}
Sender: {state['sender']}
Subject: {state['subject']}
Return ONLY valid JSON:
{{"action_plan": ["Step 1", "Step 2", "..."],"response_template": "short professional reply or null"}}
Rules:
- Max 4 steps
- If sender contains "no-reply" or "noreply" → response_template MUST be null
- For security alerts → always say "Do not click links" and "Log in directly via official site"
Return only the JSON now."""
    resp = llm_creative.invoke([HumanMessage(content=prompt)])
    try:
        cleaned = clean_json_response(resp.content)
        data = json.loads(cleaned)
        steps = data.get("action_plan", ["Review email manually"])
        template = data.get("response_template")
        if is_noreply or template in [None, "null", "None", "", "null"]:
            template = None
        return {"action_plan": steps[:4], "response_template": template}
    except Exception as e:
        return {"action_plan": ["Review this email carefully"], "response_template": None}


def update_memory(state: EmailState) -> dict:
    sender_memory.update(state["sender"], state["urgency_score"], state["risk_level"])
    return {"sender_history": sender_memory.get(state["sender"])}


# === GRAPH ===
workflow = StateGraph(EmailState)
workflow.add_node("extract", extract_email)
workflow.add_node("urgency", analyze_urgency)
workflow.add_node("expectations", detect_expectations)
workflow.add_node("risk", assess_risk)
workflow.add_node("priority", classify_priority)
workflow.add_node("actions", generate_actions)
workflow.add_node("memory", update_memory)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "urgency")
workflow.add_edge("urgency", "expectations")
workflow.add_edge("expectations", "risk")
workflow.add_edge("risk", "priority")
workflow.add_edge("priority", "actions")
workflow.add_edge("actions", "memory")
workflow.add_edge("memory", END)

app = workflow.compile()


# === DISPLAY RESULT ===
def show(state):
    print("\n" + "=" * 60)
    print("EMAIL ANALYSIS RESULT")
    print("=" * 60)
    print(f"From         : {state['sender']}")
    print(f"Subject      : {state['subject']}")
    print(f"Priority     : {state['priority']}")
    print(f"Urgency      : {state['urgency_level'].upper()} ({state['urgency_score']:.2f}/4.0)")
    print(f"Risk Level   : {state['risk_level']}")

    if state['expectations']:
        print(f"\nHidden Expectations:")
        for e in state['expectations']:
            dl = e.get('deadline') or 'no deadline'
            print(f"  • {e['description']} ({e['type']}) – {e['severity']} – {dl}")
    else:
        print(f"\nHidden Expectations: None detected")

    print(f"\nAction Plan:")
    for i, step in enumerate(state['action_plan'], 1):
        print(f"  {i}. {step}")

    print(f"\nREPLY SECTION:")
    if state['response_template']:
        print(f"→ Suggested reply:")
        print(f"  \"{state['response_template']}\"")
    else:
        print(f"→ No reply needed (no-reply address or safe to ignore)")

    # MEMORY DISPLAY
    hist = state.get("sender_history", {})
    count = hist.get("count", 0)
    if count == 0:
        print(f"\nMEMORY → First email from this sender!")
    else:
        print(f"\nMEMORY → This sender has emailed you {count} time{'s' if count != 1 else ''} before")
        print(f"   Avg urgency : {hist.get('avg_urgency', 0):.2f}/4.0")
        print(f"   High-risk   : {hist.get('high_risk_count', 0)}")
        print(f"   First seen  : {hist.get('first_seen', '')[:10]}")
        print(f"   Last seen   : {hist.get('last_seen', '')[:19]}")
    print("=" * 60)


if __name__ == "__main__":
    test_email = """ 
From: University of the People <ambassadors@uopeople.edu>
Date: Thu, Nov 13, 2025 at 12:01 PM
Subject: Share about Career Services & Virtual Internships on LinkedIn!
To: <netsanettesfaye122@gmail.com>
Alternate text
Hi Netsanet, 
Just wanted to pop in and let you know we’ve posted a brand new Ask in the Ambassador portal! 
Sincerely,
UoPeople Ambassadors Team
Check it out!
You received this email because you indicated you'd like to receive program emails.

If you don't want to receive such emails in the future, please unsubscribe 
 """
    print("Starting Smart Email Agent with PERSISTENT MEMORY...\n")
    result = app.invoke({"raw_email": test_email})
    show(result)

