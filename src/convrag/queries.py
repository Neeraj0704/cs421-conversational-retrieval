from __future__ import annotations
from .io import fingerprint

PROMPT = """Rewrite only the latest user question as one standalone search question.
Use the conversation only to resolve references, omissions, and necessary constraints.
Preserve names, exact technical terms, negation, dates, and numerical limits.
Do not answer the question or introduce facts. Assistant messages are reference-resolution
context, not verified evidence. Respect topic changes. If already standalone, keep it unchanged.
Return only the rewritten question, without an explanation."""

def latest(task):
    if not task["input"] or task["input"][-1]["speaker"] != "user":
        raise ValueError("Task must end at the latest user question")
    return task["input"][-1]["text"].strip()

def raw_history(task):
    return " ".join(x["text"] for x in task["input"] if x["speaker"]=="user").strip()

def selected_history(task):
    # Transparent heuristic, not a trained selector: keep most recent turns, preserve first topic anchor.
    turns = task["input"]
    return turns if len(turns)<=7 else [turns[0],*turns[-6:]]

def rewrite_messages(task, history_mode="full"):
    turns = task["input"] if history_mode=="full" else selected_history(task)
    history="\n".join(("USER" if x["speaker"]=="user" else "ASSISTANT")+": "+x["text"] for x in turns)
    return [{"role":"system","content":PROMPT},{"role":"user","content":history+"\nStandalone question:"}]

def cache_key(task,model,revision,history_mode):
    return fingerprint({"input":task["input"],"model":model,"revision":revision,
                        "history_mode":history_mode,"prompt":PROMPT,"max_tokens":128,"temperature":0})

def query_views(task,mode,rewrites=None):
    original=latest(task)
    if mode=="latest": return [original]
    if mode=="history": return [raw_history(task)]
    if rewrites is None or task["task_id"] not in rewrites: raise ValueError("Missing generated rewrite")
    rewrite=rewrites[task["task_id"]]
    if mode=="rewrite": return [rewrite]
    if mode=="fusion": return list(dict.fromkeys([original,rewrite]))
    if mode=="keyword":
        import re
        words=re.findall(r"[\w'-]+",rewrite)
        stop={"what","which","how","is","are","the","a","an","of","to","and","can","do","does"}
        keyword=" ".join(w for w in words if w.lower() not in stop)
        return list(dict.fromkeys([original,rewrite,keyword or rewrite]))
    raise ValueError("Unknown query mode "+mode)

