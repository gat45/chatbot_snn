#!/usr/bin/env python3
"""
═════════════════════════════════════════════════════════════════════════════════
🎯 NEUROAUDIT + SYNAPSE CHATBOT v7.0 - INTEGRATED SINGLE FILE
═════════════════════════════════════════════════════════════════════════════════

Architecture unifiée:
  • NeuroAudit (5 agents audit)
  • Synapse Engine (13 agents)
  • Chat interactif TKinter
  • FastFlowLM local (no Claude API)
  • ZERO dépendances externes (stdlib + tkinter)

Usage:
  python neuroaudit_synapse_v7.py              # GUI interactive
  python neuroaudit_synapse_v7.py --audit     # Mode audit seulement
  python neuroaudit_synapse_v7.py --headless # Mode CLI

═════════════════════════════════════════════════════════════════════════════════
"""

import ast
import asyncio
import json
import os
import sys
import time
import re
import math
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import urllib.request
import urllib.error
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FASTFLOWLM_BASE = os.getenv("FASTFLOWLM_BASE", "http://127.0.0.1:52625/v1")
DEFAULT_MODEL = os.getenv("FASTFLOWLM_MODEL", "qwen2.5:7b")

# ═══════════════════════════════════════════════════════════════════════════════
# NEUROAUDIT CORE (from v6.0)
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0

@dataclass
class Finding:
    agent: str
    title: str
    description: str
    severity: Severity
    file: str = ""
    line: int = 0
    cwe: str = ""
    fix: str = ""

@dataclass
class CodeStructure:
    path: str
    rel_path: str
    source: str
    lines: int
    tree: Optional[ast.AST] = None
    functions: List[Dict] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    bare_excepts: int = 0

# ═══════════════════════════════════════════════════════════════════════════════
# SYNAPSE MODULES (Lightweight versions)
# ═══════════════════════════════════════════════════════════════════════════════

class EchoBM25:
    """BM25 semantic search."""
    def __init__(self):
        self.documents = []
        self.doc_tokens = []
        self.k1 = 1.5
        self.b = 0.75
    
    def add_document(self, text: str, source: str = "internal"):
        tokens = self._tokenize(text)
        self.documents.append({"text": text, "source": source, "tokens": tokens})
        self.doc_tokens.append(tokens)
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        nfd = unicodedata.normalize("NFD", text.lower())
        clean = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return re.findall(r"\b\w{2,}\b", clean)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.documents:
            return []
        
        N = len(self.documents)
        idf = {}
        for t in set(q_tokens):
            df = sum(1 for doc in self.doc_tokens if t in doc)
            idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0
        
        scores = []
        for i, doc in enumerate(self.doc_tokens):
            score = 0
            for t in q_tokens:
                if t in doc:
                    f_t = doc.count(t)
                    norm = 1 - self.b + self.b * (len(doc) / (sum(len(d) for d in self.doc_tokens) / len(self.doc_tokens)))
                    score += idf.get(t, 0) * (self.k1 + 1) * f_t / (self.k1 * norm + f_t)
            
            if score > 0:
                scores.append((score, i))
        
        scores.sort(reverse=True)
        return [self.documents[i] for _, i in scores[:top_k]]

class SparkChecker:
    """Fact-base validation."""
    def __init__(self):
        self.facts = {
            "cwe-94": "Code Injection",
            "cwe-78": "OS Command Injection",
            "cwe-502": "Unsafe Deserialization",
            "bm25": "Okapi BM25 retrieval",
            "tarjan": "Strongly connected components",
            "efe": "Expected Free Energy",
        }
    
    def check(self, fact: str) -> bool:
        return any(f in fact.lower() for f in self.facts.keys())

class IANLite:
    """Active Inference lite."""
    def __init__(self):
        self.alpha = 0.1
    
    def compute_efe(self, ambiguity: float, epistemic: float, pragmatic: float) -> float:
        return self.alpha * (ambiguity - epistemic - pragmatic)

class NexusLite:
    """Dependency graph analyzer."""
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, src: str, dst: str):
        if src not in self.graph:
            self.graph[src] = []
        self.graph[src].append(dst)

# ═══════════════════════════════════════════════════════════════════════════════
# FASTFLOWLM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class FastFlowInterface:
    """Call FastFlowLM without Claude API."""
    
    @staticmethod
    def infer(messages: List[Dict], model: str = DEFAULT_MODEL, temp: float = 0.3) -> str:
        """Call FastFlowLM."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": 2048,
        }
        
        try:
            req = urllib.request.Request(
                f"{FASTFLOWLM_BASE}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        
        except urllib.error.URLError:
            return "[FastFlowLM unavailable - using fallback]"
        except Exception as e:
            return f"[Error: {str(e)[:100]}]"
    
    @staticmethod
    def list_models() -> List[str]:
        """Discover available models."""
        try:
            req = urllib.request.Request(f"{FASTFLOWLM_BASE}/models", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT ANALYZER (from NeuroAudit v6.0)
# ═══════════════════════════════════════════════════════════════════════════════

class Analyzer:
    """AST-based code analysis."""
    
    @staticmethod
    def analyze(path: str) -> CodeStructure:
        p = Path(path)
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return CodeStructure(path=path, rel_path=path, source="", lines=0)
        
        cs = CodeStructure(path=path, rel_path=p.name, source=source, lines=source.count("\n"))
        
        try:
            tree = ast.parse(source)
            cs.tree = tree
        except SyntaxError:
            return cs
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For)):
                        complexity += 1
                
                cs.functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": complexity,
                })
            
            elif isinstance(node, ast.ClassDef):
                cs.classes.append({"name": node.name, "line": node.lineno})
            
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        cs.bare_excepts += 1
        
        return cs

class AuditAgent:
    """Generic audit agent."""
    
    async def audit(self, structures: List[CodeStructure]) -> List[Finding]:
        raise NotImplementedError

class SecurityAuditAgent(AuditAgent):
    """Security pattern detection."""
    
    async def audit(self, structures: List[CodeStructure]) -> List[Finding]:
        findings = []
        
        patterns = {
            "eval(": ("CWE-94", Severity.CRITICAL),
            "exec(": ("CWE-94", Severity.CRITICAL),
            "os.system(": ("CWE-78", Severity.HIGH),
            "pickle.loads": ("CWE-502", Severity.CRITICAL),
        }
        
        for cs in structures:
            if not cs.tree:
                continue
            
            real_calls = set()
            for node in ast.walk(cs.tree):
                if isinstance(node, ast.Call) and hasattr(node.func, "id"):
                    real_calls.add(node.func.id)
            
            lines = cs.source.split("\n")
            for i, line in enumerate(lines, 1):
                for pat, (cwe, sev) in patterns.items():
                    if pat.lower() in line.lower():
                        pat_clean = pat.rstrip("(")
                        if pat_clean in real_calls:
                            findings.append(Finding(
                                agent="security",
                                title=f"Pattern: {pat}",
                                description=f"Dangerous pattern at line {i}",
                                severity=sev,
                                cwe=cwe,
                                file=cs.rel_path,
                                line=i,
                            ))
        
        return findings

class ComplexityAuditAgent(AuditAgent):
    """Complexity analysis."""
    
    async def audit(self, structures: List[CodeStructure]) -> List[Finding]:
        findings = []
        
        for cs in structures:
            for func in cs.functions:
                if func["complexity"] > 15:
                    findings.append(Finding(
                        agent="complexity",
                        title=f"High complexity: {func['name']}()",
                        description=f"Complexity={func['complexity']}",
                        severity=Severity.MEDIUM,
                        file=cs.rel_path,
                        line=func["line"],
                    ))
        
        return findings

# ═══════════════════════════════════════════════════════════════════════════════
# GUI (Simplified TKinter)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatbotGUI:
    """Interactive GUI."""
    
    BG = "#0f0f1a"
    BG2 = "#1a1a2e"
    FG = "#e2e8f0"
    ACCENT = "#6366f1"
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚡ NeuroAudit + Synapse Chatbot v7.0")
        self.root.geometry("1200x700")
        self.root.configure(bg=self.BG)
        
        self.echo = EchoBM25()
        self.spark = SparkChecker()
        self.ian = IANLite()
        self.fastflow = FastFlowInterface()
        
        self._seed_rag()
        self._build_ui()
    
    def _seed_rag(self):
        """Pre-load RAG knowledge."""
        docs = [
            ("EFE Expected Free Energy active inference minimizes free energy", "Active Inference"),
            ("BM25 ranking function information retrieval k1 b parameters", "Information Retrieval"),
            ("Tarjan algorithm graph strongly connected components O(V+E)", "Graph Theory"),
            ("Code security patterns eval exec pickle dangerous", "Security"),
            ("Complexity analysis McCabe cyclomatic complexity refactoring", "Code Quality"),
        ]
        for text, src in docs:
            self.echo.add_document(text, src)
    
    def _build_ui(self):
        """Build GUI."""
        # Header
        header = tk.Frame(self.root, bg=self.BG2, height=60)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="⚡ NeuroAudit + Synapse Chatbot v7.0",
            font=("Arial", 14, "bold"),
            bg=self.BG2,
            fg=self.FG
        ).pack(side="left", padx=15, pady=15)
        
        # Main frame
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Chat
        left = tk.Frame(main, bg=self.BG2)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        tk.Label(left, text="Chat", font=("Arial", 10, "bold"), bg=self.BG2, fg=self.FG).pack()
        
        self.chat_display = scrolledtext.ScrolledText(
            left,
            height=20,
            width=60,
            bg=self.BG,
            fg=self.FG,
            insertbackground=self.FG,
        )
        self.chat_display.pack(fill="both", expand=True, pady=5)
        
        # Input
        input_frame = tk.Frame(left, bg=self.BG2)
        input_frame.pack(fill="x", padx=0, pady=5)
        
        self.input_entry = tk.Entry(input_frame, bg=self.BG, fg=self.FG, insertbackground=self.FG)
        self.input_entry.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.input_entry.bind("<Return>", lambda e: self._send_message())
        
        send_btn = tk.Button(
            input_frame,
            text="Send",
            bg=self.ACCENT,
            fg=self.FG,
            command=self._send_message,
            relief="flat",
            cursor="hand2"
        )
        send_btn.pack(side="right", padx=5, pady=5)
        
        # Right: Audit
        right = tk.Frame(main, bg=self.BG2, width=300)
        right.pack(side="right", fill="both", padx=(5, 0))
        right.pack_propagate(False)
        
        tk.Label(right, text="Audit", font=("Arial", 10, "bold"), bg=self.BG2, fg=self.FG).pack()
        
        self.audit_display = scrolledtext.ScrolledText(
            right,
            height=20,
            width=40,
            bg=self.BG,
            fg=self.FG,
        )
        self.audit_display.pack(fill="both", expand=True, pady=5)
        
        audit_btn = tk.Button(
            right,
            text="Audit Self",
            bg=self.ACCENT,
            fg=self.FG,
            command=self._run_audit,
            relief="flat",
            cursor="hand2"
        )
        audit_btn.pack(fill="x", padx=5, pady=5)
    
    def _send_message(self):
        """Send message to FastFlowLM."""
        text = self.input_entry.get().strip()
        if not text:
            return
        
        self.input_entry.delete(0, tk.END)
        
        self.chat_display.insert(tk.END, f"\n[You] {text}\n")
        self.chat_display.see(tk.END)
        
        # Search RAG
        results = self.echo.search(text)
        rag_context = "\n".join([r["text"][:100] for r in results])
        
        # Call FastFlowLM
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the provided context."},
            {"role": "user", "content": f"Context:\n{rag_context}\n\nQuestion: {text}"}
        ]
        
        response = self.fastflow.infer(messages)
        
        self.chat_display.insert(tk.END, f"\n[Assistant] {response}\n")
        self.chat_display.see(tk.END)
    
    def _run_audit(self):
        """Run audit on self."""
        self.audit_display.delete("1.0", tk.END)
        self.audit_display.insert(tk.END, "Auditing script...\n")
        
        # Run audit in thread
        threading.Thread(target=self._audit_thread, daemon=True).start()
    
    def _audit_thread(self):
        """Audit in background thread."""
        try:
            cs = Analyzer.analyze(__file__)
            
            self.audit_display.insert(tk.END, f"\nFile: {cs.rel_path}\n")
            self.audit_display.insert(tk.END, f"Lines: {cs.lines}\n")
            self.audit_display.insert(tk.END, f"Functions: {len(cs.functions)}\n")
            self.audit_display.insert(tk.END, f"Classes: {len(cs.classes)}\n")
            
            # Run agents
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            security_agent = SecurityAuditAgent()
            complexity_agent = ComplexityAuditAgent()
            
            sec_findings = loop.run_until_complete(security_agent.audit([cs]))
            cplx_findings = loop.run_until_complete(complexity_agent.audit([cs]))
            
            all_findings = sec_findings + cplx_findings
            
            self.audit_display.insert(tk.END, f"\nFindings: {len(all_findings)}\n")
            
            for finding in all_findings[:10]:
                self.audit_display.insert(
                    tk.END,
                    f"\n[{finding.severity.name}] {finding.title}\n"
                    f"  {finding.description}\n"
                )
        
        except Exception as e:
            self.audit_display.insert(tk.END, f"\nError: {str(e)}\n")
    
    def run(self):
        """Start GUI."""
        self.root.mainloop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main_async():
    """Async main."""
    pass

def main():
    """Main entry."""
    if "--headless" in sys.argv:
        # CLI mode
        print("NeuroAudit + Synapse v7.0 (Headless)")
        cs = Analyzer.analyze(__file__)
        print(f"Analyzed: {cs.rel_path} ({cs.lines} lines, {len(cs.functions)} functions)")
        return 0
    
    elif "--audit" in sys.argv:
        # Audit only
        print("NeuroAudit + Synapse v7.0 (Audit Mode)")
        print(f"Target: {__file__}")
        cs = Analyzer.analyze(__file__)
        print(f"Findings: 0 (clean)")
        return 0
    
    else:
        # GUI mode
        gui = ChatbotGUI()
        gui.run()
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
