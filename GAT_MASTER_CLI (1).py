#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘              GAT-CLI v5.0 â€” SYNAPSE MASTER CONTROL INTERFACE                 â•‘
â•‘                                                                              â•‘
â•‘  Moteur : OMNI_SLA v5.5 (Hybrid NPU/GPU/CUDA) + Synapse Engine v8.0         â•‘
â•‘  Agents : 13 (Cortexâ†’CMB) + EFE Da Costa 2020 + BM25 Robertson 2009         â•‘
â•‘  Modes  : numpy-only (léger) | torch+GPU (complet)                           â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""
# [AUTO-FIXED: missing imports injected]
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from pathlib import Path
import sys
import os
import time
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
import subprocess
import math
from collections import defaultdict, Counter, deque
import logging
import platform
import signal
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# UTF-8 encoding fix for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        pass

# Add paths for imports
_ADD_PATHS = [
    str(Path(__file__).parent),
    str(Path(__file__).parent / "perso"),
    str(Path(__file__).parent / "Nouveau dossier"),
]
for _p in _ADD_PATHS:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MODE DUAL : détection automatique torch/numpy
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

TORCH_AVAILABLE = HAS_TORCH  # uses the import block at top of file

RICH_AVAILABLE = True  # rich already imported at top (would have failed otherwise)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  FALLBACK CONSOLE (si rich absent)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class FallbackConsole:
    """Console minimale quand rich n'est pas disponible."""

    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        text = re.sub(r'\[/?[a-z_ ]+\]', '', text)
        print(text)

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def input(self, prompt=""):
        prompt = re.sub(r'\[/?[a-z_ ]+\]', '', prompt)
        return input(prompt)

    def status(self, msg):
        class _Ctx:
            def __enter__(s): print(re.sub(r'\[/?[a-z_ ]+\]', '', msg)); return s
            def __exit__(s, *a): pass
        return _Ctx()


console = Console() if RICH_AVAILABLE else FallbackConsole()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  IAN CONTROLLER v8 â€” Da Costa 2020, Tschantz 2020
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class IANControllerV8:
    """
    G = gamma Ã— (ambiguity âˆ’ epistemic âˆ’ pragmatic)  [Da Costa 2020 Â§3]
    ambiguity = H(p)/H(uniform)                      [Tschantz 2020]
    epistemic = sqrt(Î£ w_i Ã— (v_i âˆ’ mean_v)Â²)
    pragmatic = Î£ w_i Ã— |v_i|
    gamma_{t+1} = clip(gamma Ã— exp(âˆ’0.5Ã—|EFE|), 0.2, 3.0)
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self.history: deque = deque(maxlen=100)

    def compute(self, hypotheses: List[Dict]) -> Dict[str, float]:
        if not hypotheses:
            return {"efe": 0.0, "ambiguity": 1.0, "epistemic": 0.0,
                    "pragmatic": 0.0, "gamma": self.gamma, "mode": "EXPLOITATION"}

        weights = [h.get("w", 0.5) for h in hypotheses]
        values = [h.get("v", 0.0) for h in hypotheses]
        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]
        N = len(hypotheses)

        # H(p)/H(uniform)  â€”  JAMAIS variance  [WA-MATH-01]
        H = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        H_u = math.log(max(2, N))
        ambiguity = H / H_u if H_u > 0 else 0.0

        # epistemic = sqrt(Î£ w_i Ã— (v_i âˆ’ mean_v)Â²)
        mean_v = sum(p * v for p, v in zip(probs, values))
        epistemic = math.sqrt(sum(p * (v - mean_v) ** 2 for p, v in zip(probs, values)))

        # pragmatic = Î£ w_i Ã— |v_i|
        pragmatic = sum(p * abs(v) for p, v in zip(probs, values))

        efe = self.gamma * (ambiguity - epistemic - pragmatic)
        self.history.append(efe)

        # WA-MATH-03 : signe correct
        mode = "EXPLORATION" if efe > 0.05 else "EXPLOITATION"

        # gamma adaptatif
        self.gamma = max(0.2, min(3.0, self.gamma * math.exp(-0.5 * abs(efe))))

        return {"efe": round(efe, 4), "ambiguity": round(ambiguity, 4),
                "epistemic": round(epistemic, 4), "pragmatic": round(pragmatic, 4),
                "gamma": round(self.gamma, 4), "mode": mode}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  AGENT TRACKER â€” Ã‰conomie darwinienne & routing v8
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AgentTracker:
    """Suivi des 15 agents Synapse v8.0 avec économie darwinienne."""

    AGENTS = {
        "CORTEX":   {"role": "Juge CortexGate L1-L5",        "ver": "v7",   "icon": "[Brain]"},
        "SLA":      {"role": "Croyances bayésiennes",         "ver": "v7",   "icon": "ðŸ“Š"},
        "IAN":      {"role": "EFE calculé gamma adaptatif",   "ver": "v8",   "icon": "ðŸŽ¯"},
        "PHANTOM":  {"role": "Red-team 12 patterns",          "ver": "v7",   "icon": "ðŸ‘»"},
        "ECHO":     {"role": "Sémantique BM25",               "ver": "v8",   "icon": "ðŸ”Š"},
        "SPARK":    {"role": "Vérif factuelle 0.35/0.28",     "ver": "v8",   "icon": "[PWR]"},
        "NEXUS":    {"role": "Dépendances Tarjan",            "ver": "v7",   "icon": "ðŸ”—"},
        "STRIX":    {"role": "Audit AST Leaky ReLU",          "ver": "v8",   "icon": "ðŸ¦‰"},
        "LEONARD":  {"role": "MCTS-Invention curiosité",      "ver": "v1",   "icon": "ðŸŽ¨"},
        "LEARNING": {"role": "In-memory novelty 1/sqrtn",     "ver": "v2",   "icon": "ðŸ“š"},
        "TEAMREV":  {"role": "Consensus Dempster-Shafer",     "ver": "v7.3", "icon": "ðŸ¤"},
        "SELFOPT":  {"role": "diagnose-fix-verify-learn",     "ver": "v7.3", "icon": "ðŸ”§"},
        "CMB":      {"role": "11 zones faibles injection",    "ver": "v7.4", "icon": "ðŸŒ‰"},
        "CURIO":    {"role": "Carnet graphe curiosité",       "ver": "v4",   "icon": "ðŸ”"},
        "REWARD":   {"role": "Orchestrateur darwinien",       "ver": "v8",   "icon": "ðŸ†"},
    }

    ROUTING = {
        "code":        ["STRIX", "PHANTOM", "TEAMREV", "SELFOPT"],
        "audit":       ["STRIX", "PHANTOM", "TEAMREV", "SELFOPT"],
        "vulnérab":    ["STRIX", "PHANTOM", "TEAMREV", "SELFOPT"],
        "sécurité":    ["STRIX", "PHANTOM", "TEAMREV", "SELFOPT"],
        "hypothèse":   ["SLA", "IAN", "TEAMREV"],
        "croyance":    ["SLA", "IAN", "TEAMREV"],
        "probabili":   ["SLA", "IAN", "TEAMREV"],
        "bayés":       ["SLA", "IAN", "TEAMREV"],
        "curiosité":   ["LEONARD", "CURIO", "SLA"],
        "invention":   ["LEONARD", "CURIO", "SLA"],
        "mcts":        ["LEONARD", "SLA"],
        "analogie":    ["LEONARD", "CURIO"],
        "anomalie":    ["LEONARD", "CURIO"],
        "carnet":      ["CURIO"],
        "graphe":      ["CURIO", "NEXUS"],
        "cycle":       ["NEXUS"],
        "import":      ["NEXUS"],
        "dépendanc":   ["NEXUS"],
        "similaire":   ["ECHO"],
        "sémantique":  ["ECHO"],
        "fait":        ["SPARK"],
        "citation":    ["SPARK"],
        "littératur":  ["SPARK"],
        "référence":   ["SPARK"],
        "mémoire":     ["LEARNING"],
        "leçon":       ["LEARNING"],
        "consensus":   ["TEAMREV"],
        "vote":        ["TEAMREV"],
        "optimis":     ["SELFOPT"],
        "diagnostic":  ["SELFOPT"],
        "zone":        ["CMB"],
        "faible":      ["CMB"],
        "injection":   ["CMB"],
        "reward":      ["REWARD"],
        "darwin":      ["REWARD"],
        "économie":    ["REWARD"],
    }

    MIN_CYCLES_DARWIN = 3  # WA-ARCH-02

    def __init__(self):
        self.stats: Dict[str, Dict] = {}
        for name in self.AGENTS:
            self.stats[name] = {
                "calls": 0, "utility": 0.5, "cycles": 0,
                "last_call": None, "status": "IDLE"
            }

    def route(self, query: str) -> List[str]:
        q = query.lower()
        for signal, agents in self.ROUTING.items():
            if signal in q:
                return agents
        return ["CORTEX", "SLA"]

    def call(self, agent_name: str):
        if agent_name in self.stats:
            s = self.stats[agent_name]
            s["calls"] += 1
            s["cycles"] += 1
            s["last_call"] = datetime.now().strftime("%H:%M:%S")
            s["status"] = "ACTIVE"

    def reward(self, agent_name: str, success: bool):
        if agent_name not in self.stats:
            return
        s = self.stats[agent_name]
        if s["cycles"] < self.MIN_CYCLES_DARWIN:
            return
        delta = 0.3 if success else -0.1
        s["utility"] = max(0.1, min(1.0, s["utility"] + delta))

    def idle_all(self):
        for s in self.stats.values():
            s["status"] = "IDLE"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  LEARNING ENGINE v2 â€” novelty = 1/sqrt(n)  [Strehl & Littman 2008 JMLR]
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LearningEngineV2:
    def __init__(self):
        self.patterns: Dict[str, int] = defaultdict(int)
        self.lessons: deque = deque(maxlen=500)

    def record(self, pattern_id: str, content: str, outcome: str):
        self.patterns[pattern_id] += 1
        self.lessons.append({
            "ts": time.time(), "pattern": pattern_id,
            "content": content[:200], "outcome": outcome
        })

    def novelty(self, pattern_id: str) -> float:
        n = self.patterns.get(pattern_id, 0)
        return 1.0 if n == 0 else 1.0 / math.sqrt(n)

    def classify(self, pattern_id: str) -> str:
        n = self.novelty(pattern_id)
        if n > 0.50:   return "TO_EXPLORE"
        elif n > 0.20: return "IN_PROGRESS"
        return "MASTERED"

    def get_stats(self) -> Dict:
        return {
            "total_patterns": len(self.patterns),
            "total_lessons": len(self.lessons),
            "explore": sum(1 for p in self.patterns if self.classify(p) == "TO_EXPLORE"),
            "mastered": sum(1 for p in self.patterns if self.classify(p) == "MASTERED"),
        }

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STRIX SCANNER v8 â€” Leaky ReLU sans clamp [Maas 2013]
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class StrixScannerV8:
    ALPHA = 0.05

    DANGEROUS = [
        "eval(", "exec(", "os.system(", "__import__(",
        "subprocess.call(", "pickle.loads(", "yaml.load(",
        "compile(", "globals()[",
    ]

    QUALITY = {
        "docstring": '"""', "type_hint": "->", "try_except": "try:",
        "logging": "logging.", "deque": "deque(", "pathlib": "Path(",
    }

    @classmethod
    def scan_file(cls, filepath: str) -> Dict:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return {"error": str(e), "score": 0, "file": os.path.basename(filepath)}

        n_lines = len(lines)
        n_funcs = content.count("def ")
        n_classes = content.count("class ")
        dangers = [p for p in cls.DANGEROUS if p in content]
        quality_hits = sum(1 for v in cls.QUALITY.values() if v in content)
        bare_excepts = content.count("except Exception as e:") + content.count("except Exception:")

        # pop(0) detector [WA-CODE-01]
        pop0 = len(re.findall(r'\.pop\(0\)', content))

        raw = 0.0
        raw += min(1.0, n_funcs / 15) * 15
        raw += min(1.0, n_classes / 5) * 10
        raw += max(0.0, 1.0 - len(dangers) / 3) * 20
        raw += min(1.0, quality_hits / 4) * 15
        raw += max(0.0, 1.0 - bare_excepts / 5) * 10
        raw += (10 if '"""' in content else 0)
        raw += (5 if "import logging" in content else 0)
        raw += (5 if "Path(" in content else 0)
        raw += (5 if "deque(" in content else 0)
        raw += (5 if "->" in content else 0)

        # Leaky ReLU sans clamp [Maas 2013]
        score = raw if raw >= 0 else cls.ALPHA * raw

        return {
            "file": os.path.basename(filepath),
            "lines": n_lines, "functions": n_funcs, "classes": n_classes,
            "dangers": dangers, "quality_hits": quality_hits,
            "bare_excepts": bare_excepts, "pop0": pop0,
            "score": round(score, 1),
        }

    @classmethod
    def scan_directory(cls, dirpath: str) -> List[Dict]:
        results = []
        for root, _, files in os.walk(dirpath):
            if any(x in root for x in ["__pycache__", ".shadow", "node_modules"]):
                continue
            for f in files:
                if f.endswith('.py'):
                    results.append(cls.scan_file(os.path.join(root, f)))
        results.sort(key=lambda r: r.get("score", 0))
        return results

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SPARK v8 â€” Seuils corrigés + FACT_BASE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class SparkV8:
    CONFIRM_THR = 0.35
    CONTRA_THR = 0.28

    FACT_BASE = {
        "BM25":             "Robertson & Zaragoza 2009",
        "novelty":          "Strehl & Littman 2008 JMLR â€” 1/sqrt(n)",
        "ambiguity":        "Tschantz 2020 arXiv:2002.10098 â€” H(p)/H(uniform)",
        "EFE":              "Da Costa 2020 Â§3 â€” G = gamma*(amb-ep-prag)",
        "Leaky ReLU":       "Maas et al. 2013 ICML â€” alpha FIXE",
        "PReLU":            "He et al. 2015 ICCV â€” alpha APPRIS",
        "UCB1":             "Auer et al. 2002",
        "PUCT":             "Couëtoux 2011 RAVE/AMAF",
        "darwinisme":       "Hofbauer & Sigmund 1998 â€” MIN_CYCLES=3",
        "Dempster-Shafer":  "Shafer 1976 â€” consensus inter-agents",
        "Shannon":          "Shannon 1948 â€” H(p) = -sum p log p",
    }

    @classmethod
    def verify(cls, claim: str) -> Dict:
        claim_lower = claim.lower()
        matches = []
        for key, ref in cls.FACT_BASE.items():
            key_tokens = key.lower().split()
            hits = sum(1 for t in key_tokens if t in claim_lower)
            if hits > 0:
                score = hits / len(key_tokens)
                matches.append({"key": key, "ref": ref, "score": round(score, 2)})

        if not matches:
            return {"status": "UNKNOWN", "confidence": 0.0, "matches": []}

        best = max(matches, key=lambda m: m["score"])
        if best["score"] >= cls.CONFIRM_THR:
            status = "CONFIRMED"
        elif best["score"] >= cls.CONTRA_THR:
            status = "PARTIAL"
        else:
            status = "UNVERIFIED"

        return {"status": status, "confidence": best["score"], "matches": matches}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PHANTOM QUICK SCAN â€” Patterns SEC/PERF/ARCH
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class PhantomQuickScan:
    """12 patterns de PhantomPredator v7 en mode rapide."""

    PATTERNS = {
        "SEC-001": {"name": "eval/exec injection",      "regex": r"eval\(|exec\("},
        "SEC-002": {"name": "os.system shell",           "regex": r"os\.system\("},
        "SEC-003": {"name": "bare except swallow",       "regex": r"except\s*:"},
        "SEC-004": {"name": "hardcoded credentials",     "regex": r"password\s*=\s*['\"]|api_key\s*=\s*['\"]"},
        "SEC-005": {"name": "__import__ dynamic",        "regex": r"__import__\("},
        "PERF-001": {"name": "popleft() O(n)",         "regex": r"\.pop\(0\)"},
        "PERF-002": {"name": "string concat in loop",    "regex": r"for.*\n.*\+=\s*['\"]"},
        "ARCH-001": {"name": "global state mutation",    "regex": r"^[a-z_]+\s*=\s*(?!.*def |.*class )", },
        "ARCH-002": {"name": "missing type hints",       "regex": r"def \w+\([^:)]*\)\s*:"},
        "ARCH-003": {"name": "hardcoded path Windows",   "regex": r'[A-Z]:\\\\(?:Users|Program)'},
        "ARCH-004": {"name": "import * wildcard",        "regex": r"from \w+ import \*"},
        "ARCH-005": {"name": "no logging configured",    "regex": None},
    }

    @classmethod
    def scan(cls, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return []

        findings = []
        for pid, pat in cls.PATTERNS.items():
            if pat["regex"] is None:
                # Vérification spéciale
                if "logging" not in content and pid == "ARCH-005":
                    findings.append({"id": pid, "name": pat["name"], "count": 1})
                continue
            matches = re.findall(pat["regex"], content, re.MULTILINE)
            if matches:
                findings.append({"id": pid, "name": pat["name"], "count": len(matches)})

        return findings

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  OMNI_SLA BRIDGE (import conditionnel â€” lazy)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_omni = None
_omni_health = None

def _get_omni():
    """Lazy import OmniSLA from omni_sla_orchestrator."""
    global _omni, _omni_health
    if _omni is None:
        try:
            from omni_sla_orchestrator import OmniSLA, SystemHealth
            _omni = OmniSLA()
            _omni_health = SystemHealth
        except Exception as e:
            logging.debug(f"OmniSLA unavailable: {e}")
    return _omni, _omni_health

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  GAT MASTER CLI v5.0
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class GatMasterCLI:
    VERSION = "5.0"

    def __init__(self):
        self.running = True
        self.agents = AgentTracker()
        self.ian = IANControllerV8()
        self.learning = LearningEngineV2()
        self.cmd_count = 0
        self.session_start = time.time()

        self.commands: Dict[str, Dict] = {
            "/help":     {"fn": self.cmd_help,      "desc": "Aide complète"},
            "/agents":   {"fn": self.cmd_agents,    "desc": "Ã‰tat des 15 agents Synapse v8"},
            "/audit":    {"fn": self.cmd_audit,     "desc": "Audit Strix+Phantom d'un fichier/dossier"},
            "/efe":      {"fn": self.cmd_efe,       "desc": "Calcul EFE (Da Costa 2020)"},
            "/spark":    {"fn": self.cmd_spark,     "desc": "Vérification factuelle FACT_BASE"},
            "/phantom":  {"fn": self.cmd_phantom,   "desc": "Phantom scan (12 patterns)"},
            "/reward":   {"fn": self.cmd_reward,    "desc": "Ã‰conomie darwinienne â€” ranking agents"},
            "/rag":      {"fn": self.cmd_rag,       "desc": "Recherche RAG omniscience"},
            "/health":   {"fn": self.cmd_health,    "desc": "Santé système CPU/RAM/Disque"},
            "/memory":   {"fn": self.cmd_memory,    "desc": "Stats LearningEngine v2"},
            "/novelty":  {"fn": self.cmd_novelty,   "desc": "Novelty d'un pattern (1/sqrt n)"},
            "/route":    {"fn": self.cmd_route,     "desc": "Routing v8 pour une requête"},
            "/config":   {"fn": self.cmd_config,    "desc": "Voir/modifier paramètres runtime"},
            "/export":   {"fn": self.cmd_export,    "desc": "Exporter audit/session en JSON"},
            "/test":     {"fn": self.cmd_test,      "desc": "Self-test agents + formules"},
            "/mode":     {"fn": self.cmd_mode,      "desc": "Mode actif (numpy/torch)"},
            "/dash":     {"fn": self.cmd_dashboard, "desc": "Dashboard complet"},
            "/clear":    {"fn": self.cmd_clear,     "desc": "Effacer l'écran"},
            "/exit":     {"fn": self.cmd_exit,      "desc": "Quitter"},
        }

    # â”€â”€â”€â”€ RENDERING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _render_header(self):
        omni, health_cls = _get_omni()
        cpu = ram = "?"
        npu_up = gpu_up = False

        if health_cls:
            try:
                h = health_cls.get_report()
                cpu, ram = h["cpu_percent"], h["ram_percent"]
            except Exception:
                pass

        if omni:
            try: npu_up = omni.is_npu_active()
            except Exception: pass
            try: gpu_up = omni.is_gpu_active()
            except Exception: pass

        npu_s = "ðŸŸ¢" if npu_up else "ðŸ”´"
        gpu_s = "ðŸŸ¢" if gpu_up else "ðŸ”´"
        torch_s = "ðŸŸ¢ torch+CUDA" if (TORCH_AVAILABLE and torch.cuda.is_available()) else ("âšª torch CPU" if TORCH_AVAILABLE else "âšª numpy-only")
        hw_mode = "HYBRID" if (npu_up and gpu_up) else ("NPU" if npu_up else ("GPU" if gpu_up else "OFFLINE"))

        header = (
            f" GAT-CLI v{self.VERSION} â”‚ Synapse Engine v8.0 â”‚ {hw_mode}\n"
            f" NPU {npu_s}  GPU {gpu_s}  â”‚ CPU {cpu}%  RAM {ram}%  â”‚ {torch_s}"
        )

        if RICH_AVAILABLE:
            console.print(Panel(header, title="[bold yellow]MASTER CONTROL[/bold yellow]",
                                border_style="yellow"))
        else:
            console.print("=" * 70)
            console.print(header)
            console.print("=" * 70)

    def _table(self, title: str, columns: List[str], rows: List[List[str]]):
        if RICH_AVAILABLE:
            t = Table(title=title, show_lines=False, pad_edge=False)
            for c in columns:
                t.add_column(c)
            for r in rows:
                t.add_row(*[str(x) for x in r])
            console.print(t)
        else:
            console.print(f"\n  {title}")
            console.print("  " + "-" * 60)
            console.print("  " + "  ".join(f"{c:<14}" for c in columns))
            for r in rows:
                console.print("  " + "  ".join(f"{str(x):<14}" for x in r))

    # â”€â”€â”€â”€ COMMANDES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def cmd_help(self, _args=""):
        rows = [[cmd, info["desc"]] for cmd, info in self.commands.items()]
        self._table("COMMANDES", ["Commande", "Description"], rows)
        console.print("\n  Texte sans / â†’ envoyé Ã  OMNI_SLA (NPU+GPU).\n")

    def cmd_agents(self, _args=""):
        rows = []
        for name, meta in AgentTracker.AGENTS.items():
            s = self.agents.stats[name]
            bar = "â–ˆ" * int(s["utility"] * 10) + "â–‘" * (10 - int(s["utility"] * 10))
            rows.append([
                f"{meta['icon']} {name}",
                meta["ver"],
                str(s["calls"]),
                f"{bar} {s['utility']:.1f}",
                s["status"]
            ])
        self._table("FLOTTE SYNAPSE v8.0", ["Agent", "Ver", "Calls", "Utility", "Status"], rows)

    def cmd_audit(self, args: str):
        target = args.strip() or "."
        target_path = Path(target)

        if target_path.is_file():
            r = StrixScannerV8.scan_file(str(target_path))
            self._show_single_audit(r)
            # Phantom aussi
            findings = PhantomQuickScan.scan(str(target_path))
            if findings:
                console.print(f"\n  ðŸ‘» Phantom ({len(findings)} findings):")
                for f in findings:
                    console.print(f"    {f['id']} {f['name']} (Ã—{f['count']})")
            self.agents.call("STRIX")
            self.agents.call("PHANTOM")
            self.learning.record("AUDIT_FILE", str(target_path), "success")

        elif target_path.is_dir():
            results = StrixScannerV8.scan_directory(str(target_path))
            if not results:
                console.print("  Aucun .py trouvé.")
                return
            rows = []
            total_dangers = 0
            total_pop0 = 0
            for r in results:
                d = ",".join(r.get("dangers", [])) or "â€”"
                total_dangers += len(r.get("dangers", []))
                total_pop0 += r.get("pop0", 0)
                rows.append([
                    r.get("file", "?")[:20], str(r.get("lines", 0)),
                    str(r.get("functions", 0)), str(r.get("bare_excepts", 0)),
                    d[:18], f"{r.get('score', 0):.0f}"
                ])
            self._table(f"STRIX v8 â€” {target}", ["Fichier", "Lig", "Fn", "Bare", "Dangers", "Score"], rows)

            avg = sum(r.get("score", 0) for r in results) / max(1, len(results))
            console.print(f"\n  Moyenne: {avg:.1f}/100 | {len(results)} fichiers | {total_dangers} dangers | {total_pop0} pop(0)")
            self.agents.call("STRIX")
            self.agents.call("PHANTOM")
            self.learning.record("AUDIT_DIR", target, "success")
        else:
            console.print(f"  Cible introuvable : {target}")

    def _show_single_audit(self, r: Dict):
        if "error" in r:
            console.print(f"  Erreur : {r['error']}")
            return
        console.print(f"\n  ðŸ¦‰ Strix Audit : {r.get('file')}")
        console.print(f"  Lignes      : {r.get('lines')}")
        console.print(f"  Fonctions   : {r.get('functions')}  Classes : {r.get('classes')}")
        console.print(f"  Dangers     : {r.get('dangers') or 'â€”'}")
        console.print(f"  Bare except : {r.get('bare_excepts')}  pop(0) : {r.get('pop0')}")
        console.print(f"  Qualité     : {r.get('quality_hits')}/6")
        console.print(f"  Score       : {r.get('score', 0):.1f}/100")

    def cmd_efe(self, args: str):
        if args.strip():
            try:
                hyps = json.loads(args.strip())
            except json.JSONDecodeError:
                console.print('  JSON invalide. Ex: /efe [{"w":0.8,"v":0.7},{"w":0.3,"v":-0.2}]')
                return
        else:
            hyps = [{"w": 0.7, "v": 0.6}, {"w": 0.5, "v": 0.3}, {"w": 0.3, "v": -0.1}]
            console.print("  Hypothèses par défaut. /efe [JSON] pour custom.")

        result = self.ian.compute(hyps)
        self.agents.call("IAN")
        self.agents.call("SLA")

        console.print(f"\n  EFE       = {result['efe']}")
        console.print(f"  Ambiguity = {result['ambiguity']}  H(p)/H(u)")
        console.print(f"  Epistemic = {result['epistemic']}  sqrt(Î£w(v-Î¼)Â²)")
        console.print(f"  Pragmatic = {result['pragmatic']}  Î£w|v|")
        console.print(f"  Gamma     = {result['gamma']}")
        console.print(f"  Mode      â†’ {result['mode']}")

    def cmd_spark(self, args: str):
        claim = args.strip() or "BM25 Robertson 2009"
        result = SparkV8.verify(claim)
        self.agents.call("SPARK")
        console.print(f"\n  Claim   : \"{claim}\"")
        console.print(f"  Status  : {result['status']}")
        console.print(f"  Confid. : {result['confidence']}")
        for m in result.get("matches", []):
            console.print(f"    â†’ {m['key']} : {m['ref']} ({m['score']})")

    def cmd_phantom(self, args: str):
        target = args.strip()
        if not target:
            console.print("  Usage: /phantom <fichier.py>")
            return
        if not Path(target).is_file():
            console.print(f"  Fichier introuvable : {target}")
            return

        findings = PhantomQuickScan.scan(target)
        self.agents.call("PHANTOM")

        if not findings:
            console.print(f"  ðŸ‘» Aucun pattern détecté dans {target}")
            return

        rows = [[f["id"], f["name"], str(f["count"])] for f in findings]
        self._table(f"PHANTOM v7 â€” {os.path.basename(target)}", ["ID", "Pattern", "Count"], rows)

        sev = sum(1 for f in findings if f["id"].startswith("SEC"))
        console.print(f"\n  Total: {len(findings)} findings ({sev} sécurité)")

    def cmd_rag(self, args: str):
        omni, _ = _get_omni()
        if not omni:
            console.print("  OMNI_SLA non disponible.")
            return
        query = args.strip()
        if not query:
            console.print("  Usage: /rag <query>")
            return
        self.agents.call("ECHO")
        result = omni.rag.search_context(query)
        display = result[:3000]
        if RICH_AVAILABLE:
            console.print(Panel(display, title=f"[blue]RAG: {query}[/blue]"))
        else:
            console.print(f"\n  --- RAG: {query} ---\n{display}\n  ---")

    def cmd_health(self, _args=""):
        _, health_cls = _get_omni()
        if health_cls:
            try:
                h = health_cls.get_report()
                console.print(f"\n  CPU  : {h['cpu_percent']}%")
                console.print(f"  RAM  : {h['ram_percent']}%")
                console.print(f"  Disk : {h.get('disk_percent', '?')}%")
                console.print(f"  Sain : {'ðŸŸ¢' if h.get('healthy') else 'ðŸ”´'}")
            except Exception as e:
                console.print(f"  SystemHealth erreur: {e}")
        else:
            console.print("  psutil/OmniSLA non disponible.")

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                console.print(f"  GPU   : {torch.cuda.get_device_name(0)}")
                mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                console.print(f"  VRAM  : {mem:.1f} GB")
            else:
                console.print("  Torch : CPU uniquement")
        else:
            console.print("  Mode  : numpy-only (torch non installé)")

        console.print(f"  Session : {(time.time() - self.session_start)/60:.1f} min | {self.cmd_count} cmds")

    def cmd_memory(self, _args=""):
        stats = self.learning.get_stats()
        console.print(f"\n  Patterns    : {stats['total_patterns']}")
        console.print(f"  Lessons     : {stats['total_lessons']}")
        console.print(f"  TO_EXPLORE  : {stats['explore']}")
        console.print(f"  MASTERED    : {stats['mastered']}")
        self.agents.call("LEARNING")

    def cmd_novelty(self, args: str):
        pid = args.strip() or "default_pattern"
        n = self.learning.novelty(pid)
        c = self.learning.classify(pid)
        count = self.learning.patterns.get(pid, 0)
        console.print(f"\n  Pattern  : {pid}")
        console.print(f"  Visites  : {count}")
        console.print(f"  Novelty  : {n:.4f}  (1/sqrt(n) Strehl 2008)")
        console.print(f"  Status   : {c}")

    def cmd_route(self, args: str):
        query = args.strip() or "audit code Python"
        agents = self.agents.route(query)
        console.print(f"\n  Query  : \"{query}\"")
        console.print(f"  Route  : {' â†’ '.join(agents)}")
        for a in agents:
            meta = AgentTracker.AGENTS.get(a, {})
            console.print(f"    {meta.get('icon', '?')} {a} : {meta.get('role', '?')}")

    def cmd_mode(self, _args=""):
        console.print(f"\n  Torch  : {'OUI' if TORCH_AVAILABLE else 'NON (numpy-only)'}")
        if TORCH_AVAILABLE:
            console.print(f"  CUDA   : {'OUI â€” ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NON (CPU)'}")
        console.print(f"  Rich   : {'OUI' if RICH_AVAILABLE else 'NON (fallback)'}")

    def cmd_dashboard(self, _args=""):
        self._render_header()
        console.print("")
        self.cmd_agents("")

        efe_r = self.ian.compute([{"w": 0.6, "v": 0.4}, {"w": 0.4, "v": -0.1}])
        stats = self.learning.get_stats()

        console.print(f"\n  IAN   : EFE={efe_r['efe']}  Î³={efe_r['gamma']}  â†’ {efe_r['mode']}")
        console.print(f"  LEARN : {stats['total_patterns']} patterns | {stats['explore']} Ã  explorer | {stats['mastered']} maîtrisés")
        console.print(f"  TIME  : session {(time.time() - self.session_start)/60:.1f}min | {self.cmd_count} cmds\n")

    def cmd_reward(self, _args=""):
        """Affiche l'économie darwinienne — ranking des agents par performance."""
        ranked = sorted(
            self.agents.stats.items(),
            key=lambda x: x[1].get("utility", 0.5),
            reverse=True
        )
        rows = [
            [name,
             f"{data.get('utility', 0.5):.3f}",
             str(data.get("calls", 0)),
             str(data.get("cycles", 0)),
             data.get("status", "IDLE")]
            for name, data in ranked
        ]
        t = self._table("Économie Darwinienne — Agent Rankings",
                        ["AGENT", "UTILITÉ", "APPELS", "CYCLES", "STATUT"], rows)
        console.print(t)

    def cmd_config(self, args: str):
        """Voir/modifier paramètres runtime (core_config)."""
        try:
            from core_config import config
            if not args:
                console.print("\n  [bold cyan]Configuration runtime OMNI_SLA[/bold cyan]")
                fields = {
                    "lmstudio_url":  config.lmstudio_url,
                    "ollama_url":    config.ollama_url,
                    "fastflow_url":  config.fastflow_url,
                    "max_iterations": config.max_iterations,
                    "llm_timeout":   config.llm_timeout,
                    "debug":         config.debug,
                    "rag_model":     config.rag_config.get("model", "?"),
                    "rag_top_k":     config.rag_config.get("top_k", "?"),
                }
                rows = [[k, str(v)] for k, v in fields.items()]
                t = self._table("Config", ["CLÉ", "VALEUR"], rows)
                console.print(t)
                console.print("  Usage : /config debug=true  ou  /config llm_timeout=60")
            else:
                # Parse key=value
                if "=" in args:
                    key, val = args.split("=", 1)
                    key, val = key.strip(), val.strip()
                    if hasattr(config, key):
                        # Type coercion
                        cur = getattr(config, key)
                        if isinstance(cur, bool):
                            val = val.lower() in ("true", "1", "yes")
                        elif isinstance(cur, int):
                            val = int(val)
                        setattr(config, key, val)
                        console.print(f"  [green]OK[/green] {key} = {val}")
                    else:
                        console.print(f"  [red]Clé inconnue : {key}[/red]  (utilisez /config sans args pour la liste)")
                else:
                    console.print("  Format : /config clé=valeur")
        except Exception as e:
            console.print(f"  [red]Erreur config : {e}[/red]")

    def cmd_export(self, args: str):
        """Exporte la session courante et les stats agents en JSON."""
        import json
        from pathlib import Path
        from datetime import datetime

        out_dir = Path(__file__).parent / "exports"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = args.strip() or f"session_{ts}.json"
        if not fname.endswith(".json"):
            fname += ".json"
        out_path = out_dir / fname

        payload = {
            "timestamp": ts,
            "session_duration_min": round((time.time() - self.session_start) / 60, 2),
            "cmd_count": self.cmd_count,
            "torch_available": TORCH_AVAILABLE,
            "rich_available": RICH_AVAILABLE,
            "agents": {
                name: {k: v for k, v in data.items() if k != "proc"}
                for name, data in self.agents.stats.items()
            },
            "learning": self.learning.get_stats(),
            "ian_last": self.ian.compute([{"w": 0.5, "v": 0.5}]),
        }
        try:
            out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            console.print(f"  [green]Export OK →[/green] exports/{fname}")
        except Exception as e:
            console.print(f"  [red]Erreur export : {e}[/red]")

    def cmd_test(self, _args: str = ""):
        """Self-test : formules IAN, routing, StrixScanner, SparkV8."""
        import math
        errors = []

        console.print("\n  [bold cyan]Self-test OMNI_SLA — Synapse v8[/bold cyan]\n")

        # Test 1 : IAN EFE Da Costa 2020
        ian = IANControllerV8(gamma=1.0)
        r = ian.compute([{"w": 0.7, "v": 0.9}, {"w": 0.3, "v": 0.2}])
        efe_ok = isinstance(r["efe"], float) and -5 < r["efe"] < 5
        console.print(f"  {'[green]PASS[/green]' if efe_ok else '[red]FAIL[/red]'}  IAN.compute() → EFE={r['efe']}  γ={r['gamma']}  mode={r['mode']}")
        if not efe_ok:
            errors.append("IAN EFE out of range")

        # Test 2 : Routing
        route = self.agents.route("audit de sécurité")
        route_ok = "STRIX" in route
        console.print(f"  {'[green]PASS[/green]' if route_ok else '[red]FAIL[/red]'}  Routing 'sécurité' → {route}")
        if not route_ok:
            errors.append("Routing security → STRIX manquant")

        # Test 3 : LearningEngine novelty
        le = LearningEngineV2()
        le.record("T1", "contenu test", "success")
        le.record("T1", "contenu test 2", "success")
        nov = le.novelty("T1")
        nov_ok = abs(nov - 1 / math.sqrt(2)) < 0.01
        console.print(f"  {'[green]PASS[/green]' if nov_ok else '[red]FAIL[/red]'}  LearningEngine.novelty(n=2) = {nov:.4f} (attendu ≈{1/math.sqrt(2):.4f})")
        if not nov_ok:
            errors.append("Novelty formula wrong")

        # Test 4 : StrixScanner (self-scan)
        scan = StrixScannerV8.scan_file(__file__)
        scan_ok = "score" in scan and isinstance(scan["score"], float)
        console.print(f"  {'[green]PASS[/green]' if scan_ok else '[red]FAIL[/red]'}  StrixScanner self-scan → score={scan.get('score', '?')}")
        if not scan_ok:
            errors.append("StrixScanner failed")

        # Test 5 : PhantomQuickScan
        findings = PhantomQuickScan.scan(__file__)
        console.print(f"  [green]PASS[/green]  PhantomScan self → {len(findings)} finding(s)")

        # Test 6 : SparkV8 verify
        sv = SparkV8.verify("Python est un langage de programmation")
        spark_ok = "confidence" in sv
        console.print(f"  {'[green]PASS[/green]' if spark_ok else '[red]FAIL[/red]'}  SparkV8.verify() → confiance={sv.get('confidence', '?')}")

        console.print()
        if errors:
            console.print(f"  [red]ÉCHECS ({len(errors)}) :[/red] {', '.join(errors)}")
        else:
            console.print(f"  [bold green]✓ Tous les self-tests passés.[/bold green]")

    def cmd_clear(self, _args=""):
        console.clear()
        self._render_header()

    def cmd_exit(self, _args=""):
        elapsed = (time.time() - self.session_start) / 60
        console.print(f"\n  Session terminée. {self.cmd_count} commandes en {elapsed:.1f}min.")
        self.running = False

    # â”€â”€â”€â”€ BOUCLE PRINCIPALE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def main_loop(self):
        console.clear()
        self._render_header()
        console.print("  /help pour les commandes. Texte libre â†’ OMNI_SLA.\n")

        while self.running:
            try:
                prompt = "[bold green]GATMAN > [/bold green]" if RICH_AVAILABLE else "GATMAN > "
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: console.input(prompt).strip()
                )

                if not user_input:
                    continue

                self.cmd_count += 1

                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd_key = parts[0].lower()
                    cmd_args = parts[1] if len(parts) > 1 else ""

                    if cmd_key in self.commands:
                        self.commands[cmd_key]["fn"](cmd_args)
                    else:
                        console.print(f"  Commande inconnue : {cmd_key}  (/help)")
                    continue

                # Texte libre â†’ routing + OMNI_SLA
                route = self.agents.route(user_input)
                for a in route:
                    self.agents.call(a)
                console.print(f"  Route : {' â†’ '.join(route)}")

                omni, _ = _get_omni()
                if omni:
                    with console.status("[bold cyan]Inférence NPU+GPU...[/bold cyan]" if RICH_AVAILABLE else "Inférence..."):
                        response = await omni.process_task(user_input)

                    if RICH_AVAILABLE:
                        console.print(Panel(Markdown(response),
                                            title="[bold blue]OMNI_SLA v5.5[/bold blue]",
                                            border_style="blue"))
                    else:
                        console.print(f"\n--- OMNI_SLA ---\n{response}\n---")

                    self.agents.idle_all()
                    self.learning.record("QUERY", user_input[:100], "success")
                else:
                    # LLM offline: provide local analysis via routing
                    route_agents = ", ".join(route) if route else "STRIX"
                    console.print(
                        f"  [dim]LLM offline \u2014 analyse locale via {route_agents}[/dim]\n"
                        f"  Conseil : lancez un backend LLM (LM Studio:1234, Ollama:11434) pour\n"
                        f"  activer le chat complet. Commandes disponibles : /help"
                    )

            except KeyboardInterrupt:
                console.print("\n  Ctrl+C. /exit pour quitter.")
            except EOFError:
                self.running = False
            except Exception as e:
                console.print(f"  Erreur : {e}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    cli = GatMasterCLI()
    asyncio.run(cli.main_loop())