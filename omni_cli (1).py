#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMNI_SLA CLI v4 — Interface terminale complete
Corrections bug v3: chatbot fonctionnel, RAG vue, health-check, config /config/
"""

# STDLIB
import os, sys, time, json, threading, subprocess, shutil
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict

# THIRD-PARTY (avec fallback)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich import box
from rich.markup import escape
from rich.rule import Rule

# PATHS
ROOT        = Path(__file__).parent
CONFIG_DIR  = ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
LOG_DIR     = ROOT / "logs"
RAG_DIR     = ROOT / "rag_index"
CHAT_FILE   = ROOT / "logs" / "chat_history.json"

console = Console()

# ====================================================
# CONFIG
# ====================================================
def load_config() -> dict:
    defaults = {
        "llm_servers": {
            "lmstudio": {"url": "http://127.0.0.1:1234", "enabled": True, "label": "LM Studio",
                         "chat_endpoint": "/v1/chat/completions", "models_endpoint": "/v1/models", "timeout": 60},
            "fastflow": {"url": "http://127.0.0.1:52625/v1", "enabled": True, "label": "FastFlow",
                         "chat_endpoint": "/chat/completions", "models_endpoint": "/models", "timeout": 60},
            "ollama":   {"url": "http://localhost:11434", "enabled": False, "label": "Ollama",
                         "chat_endpoint": "/api/chat", "models_endpoint": "/api/tags", "timeout": 60},
        },
        "active_llm": "lmstudio",
        "default_model": "auto",
        "rag": {"enabled": True, "index_path": "rag_index", "chunk_size": 512, "top_k": 5,
                "ignored_extensions": [".exe",".pyc",".db",".zip",".png",".jpg",".gif",".dll",".bin",".so"]},
        "agents": {"auto_start": [], "log_dir": "logs"},
        "ui": {"refresh_rate": 4, "max_log_lines": 200, "chat_history_max": 100},
    }
    CONFIG_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(defaults, indent=2, ensure_ascii=False))
        return defaults
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        return data
    except Exception as e:
        console.print(f"[red]Config invalide: {e} — valeurs par defaut[/red]")
        return defaults

CFG = load_config()

# ====================================================
# AGENTS
# ====================================================
AGENTS = {
    "synapse":  {"label":"SYNAPSE v28",     "desc":"Cerveau autonome RAG+SNN",       "script":"autonomous_synapse.py",  "icon":"🧠","color":"bright_cyan"},
    "audit":    {"label":"OMNI AUDIT",      "desc":"Audit complet securite",          "script":"omni_audit_agent.py",    "icon":"🔍","color":"bright_yellow"},
    "analyzer": {"label":"AUTO ANALYZER",   "desc":"Analyse AST Score Rapport",       "script":"auto_analyzer.py",       "icon":"📊","color":"bright_green"},
    "scanner":  {"label":"SEC SCANNER v2",  "desc":"Vulnerabilites OWASP",            "script":"omni_sec_scanner_v2.py", "icon":"🛡","color":"bright_red"},
    "fixer":    {"label":"AUTO FIXER v6",   "desc":"Correction automatique",          "script":"auto_fixer_v6.py",       "icon":"🔧","color":"bright_magenta"},
    "project":  {"label":"PROJECT ANALYZER","desc":"Vue macro dependances",           "script":"project_analyzer.py",    "icon":"🗺","color":"cyan"},
    "swarm":    {"label":"AGENT SWARM",     "desc":"Multi-agents arbitrage",          "script":"agent_swarm.py",         "icon":"🐝","color":"yellow"},
    "rag":      {"label":"KNOWLEDGE RAG",   "desc":"Recherche semantique index local","script":"knowledge_rag.py",       "icon":"📚","color":"blue"},
}
PIPELINES = {
    "full":     {"label":"PIPELINE COMPLET",   "agents":["analyzer","scanner","audit","fixer"],"icon":"⚡","desc":"Analyse->Scan->Audit->Fix"},
    "security": {"label":"PIPELINE SECURITE",  "agents":["scanner","audit"],                   "icon":"🔒","desc":"Scan->Audit securite"},
    "repair":   {"label":"PIPELINE REPARATION","agents":["analyzer","fixer"],                   "icon":"🛠","desc":"Analyse->Correction"},
}

# ====================================================
# STATE
# ====================================================
class OmniState:
    def __init__(self):
        self.logs           = deque(maxlen=CFG["ui"]["max_log_lines"])
        self.chat_history   = []
        self.running_agents = {}
        self.llm_status     = {}
        self.rag_files      = []
        self.current_view   = "dashboard"
        self.lock           = threading.Lock()
        self.llm_thinking   = False
        self.active_llm     = CFG.get("active_llm","lmstudio")

    def log(self, msg:str, level:str="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO":"dim white","OK":"bright_green","ERR":"bright_red",
                  "WARN":"bright_yellow","RUN":"bright_cyan","SYS":"magenta",
                  "CHAT":"bright_blue","RAG":"blue","CFG":"green"}
        self.logs.append((ts, level, str(msg), colors.get(level,"white")))

    def add_chat(self, role:str, content:str, model:str=""):
        entry = {"role":role,"content":content,"ts":datetime.now().isoformat(),"model":model}
        self.chat_history.append(entry)
        if len(self.chat_history) > CFG["ui"]["chat_history_max"]:
            self.chat_history.pop(0)
        self._save_chat()

    def _save_chat(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            CHAT_FILE.write_text(json.dumps(self.chat_history, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def load_chat(self):
        try:
            if CHAT_FILE.exists():
                self.chat_history = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        except Exception:
            self.chat_history = []

    def start_agent(self, key:str, proc):
        with self.lock:
            self.running_agents[key] = {"proc":proc,"start":time.time(),"status":"RUNNING"}

    def update_agents(self):
        with self.lock:
            for key, info in self.running_agents.items():
                if info["proc"] and info["proc"].poll() is not None:
                    rc = info["proc"].returncode
                    if info["status"] == "RUNNING":
                        info["status"] = "✅ DONE" if rc==0 else f"❌ ERR({rc})"

STATE = OmniState()

# ====================================================
# HEALTH CHECK
# ====================================================
def check_llm(name:str, srv_cfg:dict) -> dict:
    if not HAS_REQUESTS:
        return {"ok":False,"latency_ms":0,"model":"","error":"requests manquant"}
    url = srv_cfg["url"].rstrip("/")
    ep  = srv_cfg.get("models_endpoint","/v1/models")
    t0  = time.time()
    try:
        resp = requests.get(f"{url}{ep}", timeout=srv_cfg.get("timeout",5))
        lat  = int((time.time()-t0)*1000)
        if resp.status_code == 200:
            data   = resp.json()
            models = []
            if "data" in data:
                models = [m.get("id","") for m in data["data"]]
            elif "models" in data:
                models = [m.get("name",m.get("id","")) for m in data["models"]]
            return {"ok":True,"latency_ms":lat,"model":models[0] if models else "?","models":models,"error":""}
        return {"ok":False,"latency_ms":lat,"model":"","error":f"HTTP {resp.status_code}"}
    except Exception as e:
        err = str(e)
        if "Connection" in err or "refused" in err.lower():
            return {"ok":False,"latency_ms":0,"model":"","error":"OFFLINE"}
        return {"ok":False,"latency_ms":0,"model":"","error":err[:40]}

def run_health_check():
    def _check():
        STATE.log("Health-check complet demarrage...", "SYS")
        for name, srv in CFG["llm_servers"].items():
            if not srv.get("enabled",False):
                STATE.llm_status[name] = {"ok":False,"error":"DISABLED"}
                continue
            STATE.log(f"  Check {srv['label']} ({srv['url']})...", "INFO")
            result = check_llm(name, srv)
            STATE.llm_status[name] = result
            if result["ok"]:
                STATE.log(f"  OK {srv['label']} - {result['model']} ({result['latency_ms']}ms)","OK")
            else:
                STATE.log(f"  KO {srv['label']} - {result['error']}","WARN")
        # RAG
        if RAG_DIR.exists():
            files = [str(f.relative_to(RAG_DIR)) for f in RAG_DIR.rglob("*") if f.is_file()]
            STATE.rag_files = files
            STATE.log(f"  RAG: {len(files)} fichiers indexes","RAG")
        else:
            STATE.log("  RAG: dossier rag_index absent","WARN")
        # Scripts
        py_ok = sum(1 for p in ROOT.glob("*.py") if p.stat().st_size > 0)
        STATE.log(f"  {py_ok} scripts Python trouves","SYS")
        STATE.log("Health-check termine","OK")
        # Auto-start
        for svc in CFG["agents"].get("auto_start",[]):
            if svc in AGENTS:
                STATE.log(f"  Auto-start: {AGENTS[svc]['label']}","RUN")
                launch_agent(svc)
    threading.Thread(target=_check, daemon=True).start()

# ====================================================
# LLM CHAT
# ====================================================
def get_active_srv() -> Optional[dict]:
    return CFG["llm_servers"].get(STATE.active_llm)

def send_to_llm(message:str) -> str:
    if not HAS_REQUESTS:
        return "ERREUR: 'requests' non installe. pip install requests"
    srv = get_active_srv()
    if not srv:
        return f"ERREUR: LLM '{STATE.active_llm}' non configure dans /config/settings.json"
    status = STATE.llm_status.get(STATE.active_llm,{})
    if not status.get("ok"):
        err = status.get("error","OFFLINE")
        return (f"ERREUR: {srv['label']} non disponible ({err})\n"
                f"  URL: {srv['url']}\n"
                f"  Verifiez que le serveur est demarré\n"
                f"  Config: /config/settings.json")
    url      = srv["url"].rstrip("/")
    endpoint = srv.get("chat_endpoint","/v1/chat/completions")
    model    = status.get("model", CFG.get("default_model",""))
    timeout  = srv.get("timeout",60)
    msgs = []
    for entry in STATE.chat_history[-10:]:
        msgs.append({"role":entry["role"],"content":entry["content"]})
    msgs.append({"role":"user","content":message})
    payload = {"messages":msgs,"temperature":0.7,"max_tokens":2048,"stream":False}
    if model and model not in ("auto","?",""):
        payload["model"] = model
    try:
        resp = requests.post(f"{url}{endpoint}", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message",{}).get("content","")
            if content:
                return content.strip()
        if "message" in data:
            return data["message"].get("content","").strip()
        return f"Reponse inattendue: {str(data)[:200]}"
    except requests.exceptions.Timeout:
        return f"TIMEOUT ({timeout}s) — le LLM est trop lent ou le message trop long"
    except requests.exceptions.ConnectionError:
        return f"CONNEXION PERDUE avec {srv['label']} ({url})"
    except Exception as e:
        return f"ERREUR {type(e).__name__}: {str(e)[:200]}"

def chat_with_llm(message:str):
    STATE.llm_thinking = True
    STATE.add_chat("user", message)
    STATE.log(f"Envoi au LLM: {message[:50]}...","CHAT")
    def _send():
        reply = send_to_llm(message)
        model = STATE.llm_status.get(STATE.active_llm,{}).get("model","")
        STATE.add_chat("assistant", reply, model=model)
        STATE.llm_thinking = False
        STATE.log(f"Reponse LLM recue ({len(reply)} chars)","CHAT")
    threading.Thread(target=_send, daemon=True).start()

# ====================================================
# RAG
# ====================================================
def index_path(path_str:str):
    src = Path(path_str.strip().strip('"').strip("'"))
    if not src.exists():
        STATE.log(f"RAG: chemin introuvable: {src}","ERR")
        return
    RAG_DIR.mkdir(exist_ok=True)
    ignored = set(CFG["rag"].get("ignored_extensions",[]))
    if src.is_file():
        dest = RAG_DIR / src.name
        shutil.copy2(src, dest)
        STATE.rag_files.append(src.name)
        STATE.log(f"RAG: fichier indexe: {src.name}","RAG")
    elif src.is_dir():
        dest_dir = RAG_DIR / src.name
        dest_dir.mkdir(exist_ok=True)
        count = 0
        for f in src.rglob("*"):
            if f.is_file() and f.suffix not in ignored:
                rel = f.relative_to(src)
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                STATE.rag_files.append(str(Path(src.name)/rel))
                count += 1
        STATE.log(f"RAG: dossier '{src.name}' -> {count} fichiers","RAG")

def search_rag(query:str) -> List[dict]:
    results = []
    words = query.lower().split()
    if not RAG_DIR.exists():
        return []
    for f in RAG_DIR.rglob("*"):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            tl   = text.lower()
            score = sum(tl.count(w) for w in words)
            if score > 0:
                idx     = tl.find(words[0]) if words else 0
                context = text[max(0,idx-60):idx+160].replace("\n"," ")
                results.append({"file":str(f.relative_to(RAG_DIR)),"score":score,"context":context})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:CFG["rag"].get("top_k",5)]

# ====================================================
# METRICS
# ====================================================
def get_metrics() -> dict:
    if not HAS_PSUTIL:
        return {"cpu":0,"ram":0,"ram_used":0,"ram_total":0,"disk":0}
    cpu  = psutil.cpu_percent(interval=None)
    vm   = psutil.virtual_memory()
    disk = psutil.disk_usage("/").percent if os.name!="nt" else psutil.disk_usage("C:\\").percent
    return {"cpu":cpu,"ram":vm.percent,"ram_used":vm.used//1024//1024,"ram_total":vm.total//1024//1024,"disk":disk}

def bar(val:float,width:int=12)->str:
    filled=int(val/100*width)
    return "█"*filled+"░"*(width-filled)

def mc(val:float)->str:
    return "bright_green" if val<50 else ("bright_yellow" if val<80 else "bright_red")

# ====================================================
# PANELS
# ====================================================
def make_header() -> Panel:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    srv = CFG["llm_servers"].get(STATE.active_llm,{})
    st  = STATE.llm_status.get(STATE.active_llm,{})
    llm_txt = f"[bright_green]◉ {srv.get('label',STATE.active_llm)}[/]" if st.get("ok") else f"[dim red]◯ {srv.get('label',STATE.active_llm)}[/]"
    views = {"dashboard":"📊 DASHBOARD","chat":"💬 CHAT LLM","rag":"📚 RAG","config":"⚙ CONFIG"}
    t = Text()
    t.append("◈ OMNI", style="bold white"); t.append("-SLA", style="bold bright_cyan")
    t.append("  v4  ", style="dim cyan"); t.append(f"  {now}", style="dim white")
    t.append(f"  ·  LLM: "); t.append(llm_txt)
    t.append(f"  ·  {views.get(STATE.current_view,'')}", style="bold yellow")
    return Panel(Align.center(t), style="cyan", padding=(0,2), height=3)

def make_agents_panel() -> Panel:
    STATE.update_agents()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim white", expand=True, padding=(0,1))
    table.add_column("#",style="bold dim",width=3)
    table.add_column("AGENT",style="bold",min_width=20)
    table.add_column("DESCRIPTION",style="dim white",min_width=32)
    table.add_column("STATUT",justify="right",min_width=14)
    for i,(key,agent) in enumerate(AGENTS.items(),1):
        info = STATE.running_agents.get(key)
        if info:
            s,elapsed = info["status"],time.time()-info["start"]
            if s=="RUNNING":
                sp = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.time()*8)%10]
                stxt = Text(f"{sp} {elapsed:.0f}s",style="bright_cyan bold")
            else:
                stxt = Text(s,style="bright_green" if "DONE" in s else "bright_red")
        else:
            exists = (ROOT/agent["script"]).exists()
            stxt   = Text("● PRÊT" if exists else "○ ABSENT",style="dim green" if exists else "dim red")
        label = Text(); label.append(f"{agent['icon']} "); label.append(agent["label"],style=agent["color"]+" bold")
        table.add_row(Text(str(i),style="bold yellow"),label,agent["desc"],stxt)
    return Panel(table,title="[bold cyan]◈ AGENTS[/bold cyan]",border_style="cyan",padding=(0,1))

def make_pipelines_panel() -> Panel:
    table = Table(box=box.SIMPLE,show_header=False,expand=True,padding=(0,1))
    table.add_column("",width=4); table.add_column("PIPELINE",style="bold",min_width=22); table.add_column("SÉQUENCE")
    km = {"full":"F1","security":"F2","repair":"F3"}
    for key,pipe in PIPELINES.items():
        seq=Text()
        for j,a in enumerate(pipe["agents"]):
            ag=AGENTS[a]; seq.append(ag["icon"]+" "+ag["label"],style=ag["color"])
            if j<len(pipe["agents"])-1: seq.append(" → ",style="dim")
        table.add_row(Text(km[key],style="bold yellow"),Text(f"{pipe['icon']} {pipe['label']}",style="bold white"),seq)
    return Panel(table,title="[bold yellow]◈ PIPELINES[/bold yellow]",border_style="yellow",padding=(0,1),height=9)

def make_metrics_panel() -> Panel:
    m=get_metrics(); t=Text()
    for label,val in [("CPU ",m["cpu"]),("RAM ",m["ram"]),("DISK",m["disk"])]:
        c=mc(val); t.append(f"{label} ",style="dim white"); t.append(f"[{bar(val)}]",style=c); t.append(f" {val:5.1f}%\n",style=c+" bold")
    t.append(f"     {m['ram_used']}MB/{m['ram_total']}MB\n\n",style="dim")
    active=sum(1 for v in STATE.running_agents.values() if v["status"]=="RUNNING")
    t.append("AGENTS: ",style="dim white"); t.append(f"{active}/{len(AGENTS)}",style="bright_cyan bold" if active else "dim"); t.append(" actifs\n\n",style="dim")
    t.append("LLM SERVERS:\n",style="dim bold")
    for name,srv in CFG["llm_servers"].items():
        if not srv.get("enabled",False): continue
        st=STATE.llm_status.get(name,{})
        if st.get("ok"):
            t.append(f"  ◉ {srv['label']:<12}",style="bright_green"); t.append(f"{st.get('model','?')[:18]} {st.get('latency_ms',0)}ms\n",style="dim green")
        else:
            err=st.get("error","..."); t.append(f"  ◯ {srv['label']:<12}",style="bright_red"); t.append(f"{err[:20]}\n",style="dim red")
    return Panel(t,title="[bold magenta]◈ SYSTÈME[/bold magenta]",border_style="magenta",padding=(0,1))

def make_log_panel() -> Panel:
    t=Text()
    for ts,level,msg,color in list(STATE.logs)[-16:]:
        t.append(f"{ts} ",style="dim"); t.append(f"[{level:4s}] ",style=color+" bold"); t.append(escape(str(msg)[:72])+"\n",style="white")
    if not STATE.logs: t.append("En attente...\n",style="dim italic")
    return Panel(t,title="[bold white]◈ LOG[/bold white]",border_style="white",padding=(0,1))

def make_help_panel() -> Panel:
    t=Text()
    cmds=[("1-8","Lancer un agent"),("F1/F2/F3","Pipelines"),("chat","Vue chatbot LLM"),("rag","Vue RAG + import"),
          ("cfg","Voir config"),("check","Health-check"),("llm <nom>","Changer LLM"),("kill","Stopper tous"),("q","Quitter")]
    for i,(k,d) in enumerate(cmds):
        t.append(f"  {k:<12}",style="bold bright_cyan"); t.append(d,style="dim white"); t.append("   " if i%2==0 else "\n")
    return Panel(t,title="[bold dim]◈ COMMANDES[/bold dim]",border_style="dim",padding=(0,1))

# ── Vue Chat ────────────────────────────────────────────
def make_chat_panel() -> Panel:
    t=Text()
    msgs = STATE.chat_history[-18:]
    if not msgs:
        t.append("Aucun message. Tapez votre question.\n\n",style="dim italic")
        srv=get_active_srv(); st=STATE.llm_status.get(STATE.active_llm,{})
        t.append("LLM: ",style="dim")
        if st.get("ok"):
            t.append(f"✅ {srv.get('label','')} - {st.get('model','?')}\n",style="bright_green")
        else:
            t.append(f"❌ {srv.get('label','')} OFFLINE — {st.get('error','')}\n",style="bright_red")
        t.append("\n[back]=dashboard  [clear]=effacer  [llm <nom>]=changer LLM\n",style="dim")
        return Panel(t,title="[bold blue]💬 CHAT LLM[/bold blue]",border_style="blue",padding=(0,2))
    for msg in msgs:
        role=msg.get("role","user"); content=msg.get("content","")
        ts=msg.get("ts","")[11:16]; model=msg.get("model","")
        if role=="user":
            t.append(f"\n[{ts}] ",style="dim"); t.append("▶ VOUS\n",style="bold bright_cyan")
            # Affichage complet du message utilisateur
            for line in content.split("\n"):
                t.append(f"  {escape(line)}\n",style="white")
        else:
            t.append(f"\n[{ts}] ",style="dim"); t.append("🤖 IA",style="bold bright_blue")
            if model: t.append(f" ({model[:28]})",style="dim blue")
            t.append("\n")
            # Affichage complet de la réponse IA — CORRECTION BUG: texte tronqué
            for line in content.split("\n"):
                t.append(f"  {escape(line)}\n",style="bright_white")
    if STATE.llm_thinking:
        sp="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.time()*5)%10]
        t.append(f"\n🤖 IA reflechit {sp}\n",style="bold bright_blue")
    return Panel(t,title="[bold blue]💬 CHAT LLM — Historique[/bold blue]",border_style="blue",padding=(0,2))

def make_chat_side() -> Panel:
    t=Text(); srv=get_active_srv() or {}; st=STATE.llm_status.get(STATE.active_llm,{})
    t.append("LLM ACTIF:\n",style="bold dim")
    if st.get("ok"):
        t.append(f"✅ {srv.get('label','')}\n",style="bright_green bold")
        t.append(f"   {st.get('model','?')[:30]}\n",style="dim green")
        t.append(f"   {st.get('latency_ms',0)}ms\n\n",style="dim green")
    else:
        t.append(f"❌ {srv.get('label','')} {st.get('error','')}\n\n",style="bright_red bold")
    t.append("COMMANDES:\n",style="bold dim")
    for k,d in [("back","Dashboard"),("clear","Effacer"),("check","Re-check LLM"),
                ("llm lmstudio","LM Studio"),("llm fastflow","FastFlow"),("llm ollama","Ollama"),
                ("rag <query>","Chercher RAG"),("q","Quitter")]:
        t.append(f"  {k:<15}",style="bright_cyan bold"); t.append(f"{d}\n",style="dim white")
    return Panel(t,title="[bold dim]◈ CHAT STATUS[/bold dim]",border_style="dim blue",padding=(0,1))

# ── Vue RAG ─────────────────────────────────────────────
def make_rag_panel() -> Panel:
    t=Text(); t.append(f"Index: {RAG_DIR}\n",style="dim")
    t.append(f"Fichiers: {len(STATE.rag_files)}\n\n",style="bright_cyan")
    if STATE.rag_files:
        for i,f in enumerate(STATE.rag_files[-28:],1):
            t.append(f"  {i:3}. ",style="dim"); t.append(f"{f}\n",style="blue")
    else:
        t.append("Aucun fichier indexe.\n\n",style="dim italic")
        t.append("add <chemin_fichier_ou_dossier>\n",style="bright_cyan")
        t.append("Exemple: add /home/user/docs\n",style="dim")
        t.append('Exemple Windows: add "C:\\Users\\user\\docs"\n',style="dim")
    return Panel(t,title="[bold blue]📚 RAG — Fichiers indexes[/bold blue]",border_style="blue",padding=(0,2))

def make_rag_side() -> Panel:
    t=Text(); t.append("COMMANDES RAG:\n",style="bold dim")
    for k,d in [("add <chemin>","Indexer fichier/dossier"),("search <query>","Rechercher"),
                ("list","Rafraichir liste"),("back","Dashboard"),("q","Quitter")]:
        t.append(f"  {k:<20}",style="bright_cyan bold"); t.append(f"{d}\n",style="dim white")
    t.append("\nCONFIG (/config/settings.json):\n",style="dim bold")
    rc=CFG.get("rag",{})
    t.append(f"  chunk_size: {rc.get('chunk_size',512)}\n",style="dim")
    t.append(f"  top_k: {rc.get('top_k',5)}\n",style="dim")
    ignored=rc.get("ignored_extensions",[])[:5]
    t.append(f"  ignore: {' '.join(ignored)}\n",style="dim")
    return Panel(t,title="[bold dim]◈ RAG HELP[/bold dim]",border_style="dim blue",padding=(0,1))

# ====================================================
# LAYOUT
# ====================================================
def build_layout() -> Layout:
    layout = Layout()
    if STATE.current_view=="chat":
        layout.split_column(Layout(name="h",size=3),Layout(name="b"),Layout(name="f",size=3))
        layout["b"].split_row(Layout(name="cm",ratio=3),Layout(name="cs",ratio=1))
        layout["h"].update(make_header()); layout["cm"].update(make_chat_panel())
        layout["cs"].update(make_chat_side())
        layout["f"].update(Panel(Text("  Tapez votre message puis Entree  |  back=dashboard  |  clear  |  q=quitter",style="dim"),style="blue",height=3))
    elif STATE.current_view=="rag":
        layout.split_column(Layout(name="h",size=3),Layout(name="b"),Layout(name="f",size=3))
        layout["b"].split_row(Layout(name="rm",ratio=3),Layout(name="rs",ratio=1))
        layout["h"].update(make_header()); layout["rm"].update(make_rag_panel()); layout["rs"].update(make_rag_side())
        layout["f"].update(Panel(Text("  add <chemin>  |  search <query>  |  list  |  back  |  q=quitter",style="dim"),style="blue",height=3))
    elif STATE.current_view=="config":
        cfg_txt=Text(); cfg_txt.append(f"Fichier: {CONFIG_FILE}\n\n",style="dim")
        cfg_txt.append(json.dumps(CFG,indent=2,ensure_ascii=False)[:4000],style="green")
        layout.split_column(Layout(name="h",size=3),Layout(name="b"),Layout(name="f",size=3))
        layout["h"].update(make_header())
        layout["b"].update(Panel(cfg_txt,title="[bold green]⚙ CONFIG[/bold green]",border_style="green",padding=(0,2)))
        layout["f"].update(Panel(Text("  Editez /config/settings.json  |  check=recharger  |  back=dashboard",style="dim"),style="green",height=3))
    else:  # dashboard
        layout.split_column(Layout(name="h",size=3),Layout(name="b"),Layout(name="f",size=5))
        layout["b"].split_row(Layout(name="l",ratio=3),Layout(name="r",ratio=2))
        layout["l"].split_column(Layout(name="agents"),Layout(name="pipelines",size=9))
        layout["r"].split_column(Layout(name="metrics"),Layout(name="log"))
        layout["h"].update(make_header()); layout["agents"].update(make_agents_panel())
        layout["pipelines"].update(make_pipelines_panel()); layout["metrics"].update(make_metrics_panel())
        layout["log"].update(make_log_panel()); layout["f"].update(make_help_panel())
    return layout

# ====================================================
# LAUNCH
# ====================================================
def launch_agent(key:str)->bool:
    if key not in AGENTS: STATE.log(f"Agent inconnu: {key}","ERR"); return False
    agent=AGENTS[key]; script=ROOT/agent["script"]
    if not script.exists(): STATE.log(f"Script manquant: {agent['script']}","ERR"); return False
    try:
        LOG_DIR.mkdir(exist_ok=True)
        lf=LOG_DIR/f"{key}_{datetime.now().strftime('%H%M%S')}.log"
        fh=open(lf,"w"); proc=subprocess.Popen([sys.executable,str(script)],stdout=fh,stderr=subprocess.STDOUT,cwd=str(ROOT)); fh.close()
        STATE.start_agent(key,proc); STATE.log(f"{agent['icon']} {agent['label']} PID {proc.pid}","RUN"); return True
    except Exception as e: STATE.log(f"Erreur lancement {key}: {e}","ERR"); return False

def launch_pipeline(pk:str):
    if pk not in PIPELINES: return
    pipe=PIPELINES[pk]; STATE.log(f"{pipe['icon']} Pipeline {pipe['label']}","RUN")
    for ak in pipe["agents"]: launch_agent(ak); time.sleep(0.3)

def kill_all():
    count=0
    for key,info in STATE.running_agents.items():
        if info["status"]=="RUNNING" and info["proc"]:
            try: info["proc"].terminate(); info["status"]="🛑 KILLED"; count+=1
            except: pass
    STATE.log(f"🛑 {count} agent(s) arretes","WARN")

# ====================================================
# COMMAND HANDLER
# ====================================================
SHORTCUTS = {str(i):key for i,key in enumerate(AGENTS.keys(),1)}
SHORTCUTS.update({"f1":"full","f2":"security","f3":"repair"})

def handle_command(cmd:str)->bool:
    raw=cmd.strip(); cmd=raw.lower()
    if not raw: return True
    if cmd in ("q","quit","exit"): return False

    # Navigation
    if cmd in ("chat","c"):
        STATE.current_view="chat"; STATE.log("Vue CHAT — tapez votre message","SYS"); return True
    if cmd in ("rag","r"):
        STATE.current_view="rag"; STATE.log("Vue RAG","RAG"); return True
    if cmd in ("cfg","config"):
        global CFG; CFG=load_config(); STATE.current_view="config"; STATE.log("Vue CONFIG","CFG"); return True
    if cmd in ("back","dashboard","home","d"):
        STATE.current_view="dashboard"; return True

    # Vue Chat — tout message non-commande va au LLM
    if STATE.current_view=="chat":
        if cmd=="clear": STATE.chat_history.clear(); STATE.log("Chat efface","SYS"); return True
        if cmd=="check": run_health_check(); return True
        if cmd.startswith("llm "):
            nl=cmd[4:].strip()
            if nl in CFG["llm_servers"]:
                STATE.active_llm=nl; CFG["active_llm"]=nl
                try: CONFIG_FILE.write_text(json.dumps(CFG,indent=2,ensure_ascii=False))
                except: pass
                STATE.log(f"LLM -> {nl}","SYS")
                res=check_llm(nl,CFG["llm_servers"][nl]); STATE.llm_status[nl]=res
            else:
                STATE.log(f"LLM inconnu. Disponibles: {', '.join(CFG['llm_servers'].keys())}","ERR")
            return True
        if cmd.startswith("rag "):
            query=raw[4:].strip(); results=search_rag(query)
            STATE.log(f"RAG '{query}': {len(results)} resultats","RAG")
            for r in results: STATE.log(f"  [{r['score']}] {r['file']}: {r['context'][:50]}","RAG")
            return True
        # Message au LLM (commande par défaut en mode chat)
        chat_with_llm(raw)
        return True

    # Vue RAG
    if STATE.current_view=="rag":
        if cmd.startswith("add "): index_path(raw[4:].strip()); return True
        if cmd.startswith("search "): 
            query=raw[7:].strip(); results=search_rag(query)
            STATE.log(f"RAG search '{query}': {len(results)} resultats","RAG")
            for r in results: STATE.log(f"  [{r['score']}] {r['file']}: {r['context'][:60]}","RAG")
            return True
        if cmd=="list":
            if RAG_DIR.exists():
                files=[str(f.relative_to(RAG_DIR)) for f in RAG_DIR.rglob("*") if f.is_file()]
                STATE.rag_files=files; STATE.log(f"RAG: {len(files)} fichiers","RAG")
            return True

    # Commandes globales
    if cmd in ("clear","cls"): STATE.logs.clear(); STATE.log("Logs effaces","SYS"); return True
    if cmd=="status":
        STATE.update_agents()
        if not STATE.running_agents: STATE.log("Aucun agent lance","SYS")
        for key,info in STATE.running_agents.items():
            STATE.log(f"  {AGENTS[key]['icon']} {AGENTS[key]['label']}: {info['status']} ({time.time()-info['start']:.0f}s)","SYS")
        return True
    if cmd=="kill": kill_all(); return True
    if cmd=="check": run_health_check(); return True
    if cmd.startswith("llm "):
        nl=cmd[4:].strip()
        if nl in CFG["llm_servers"]:
            STATE.active_llm=nl; CFG["active_llm"]=nl; STATE.log(f"LLM -> {nl}","SYS")
        else: STATE.log(f"LLM inconnu. Dispo: {', '.join(CFG['llm_servers'].keys())}","ERR")
        return True
    if cmd in SHORTCUTS:
        t=SHORTCUTS[cmd]
        if t in AGENTS: launch_agent(t)
        elif t in PIPELINES: launch_pipeline(t)
        return True
    if cmd in AGENTS: launch_agent(cmd); return True
    if cmd in PIPELINES: launch_pipeline(cmd); return True

    STATE.log(f"Commande inconnue: '{cmd}' -> tapez 1-8, chat, rag, cfg, check, q","WARN")
    return True

# ====================================================
# MAIN LOOP — BUG FIX: input non-bloquant avec Live
# ====================================================
def main_loop():
    """
    CORRECTION BUG v3:
    Le problème était que console.input() dans un contexte Live
    écrase l'affichage et rend l'input invisible.
    
    Solution: Thread séparé pour l'input, Live rafraîchi en parallèle.
    Le Live s'arrête avant l'affichage de la réponse, puis reprend.
    """
    LOG_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    STATE.load_chat()

    console.print(Rule("[bold cyan]◈ OMNI-SLA CLI v4[/bold cyan]"))
    STATE.log("OMNI-SLA CLI v4 demarre","SYS")
    STATE.log(f"Config: {CONFIG_FILE}","CFG")
    STATE.log(f"LLM actif: {STATE.active_llm} -> {CFG['llm_servers'].get(STATE.active_llm,{}).get('url','')}","CFG")
    STATE.log(f"{len(list(ROOT.glob('*.py')))} scripts Python dans {ROOT}","SYS")
    run_health_check()

    running = True
    while running:
        cmd_holder = []
        done_event = threading.Event()

        def get_input():
            prompts = {
                "dashboard": "\n[bold cyan]  ◈[/bold cyan] [dim]COMMAND >[/dim] ",
                "chat":      "\n[bold blue]  💬[/bold blue] [dim]MESSAGE >[/dim] ",
                "rag":       "\n[bold blue]  📚[/bold blue] [dim]RAG CMD >[/dim] ",
                "config":    "\n[bold green]  ⚙[/bold green]  [dim]CMD >[/dim] ",
            }
            try:
                val = console.input(prompts.get(STATE.current_view, "\n  ◈ > "))
                cmd_holder.append(val)
            except (EOFError, KeyboardInterrupt):
                cmd_holder.append("q")
            finally:
                done_event.set()

        t = threading.Thread(target=get_input, daemon=True)
        t.start()

        # Live rafraîchi pendant l'attente de l'input
        with Live(build_layout(), refresh_per_second=CFG["ui"].get("refresh_rate",4),
                  screen=True, console=console) as live:
            while not done_event.wait(timeout=0.25):
                live.update(build_layout())
                STATE.update_agents()
            live.update(build_layout())

        cmd = cmd_holder[0] if cmd_holder else "q"
        running = handle_command(cmd)

    kill_all()
    console.print("\n[bold cyan]◈ OMNI-SLA v4 — A bientot.[/bold cyan]\n")

if __name__ == "__main__":
    main_loop()
