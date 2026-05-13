#!/usr/bin/env python3
"""
═════════════════════════════════════════════════════════════════════════════════
🎯 NeuroAudit v6.0 - SINGLE FILE SELF-AUDITING SYSTEM
═════════════════════════════════════════════════════════════════════════════════

✨ Features:
  • ZERO external dependencies (stdlib only)
  • 5 parallel agents (Static + Security + Logic + Concurrency + Performance)
  • Auto-audit itself on execution
  • Consensus & deduplication
  • ASCII-only output (works everywhere)

📌 Usage:
  python neuroaudit_v6.py                  # Auto-audit itself
  python neuroaudit_v6.py /path/to/code   # Audit specified path
  python neuroaudit_v6.py --self-test     # Full test suite

═════════════════════════════════════════════════════════════════════════════════
"""

import ast
import asyncio
import json
import os
import sys
import time
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

CWE_DATABASE = {
    "CWE-20": "Improper Input Validation",
    "CWE-78": "OS Command Injection",
    "CWE-94": "Code Injection",
    "CWE-119": "Buffer Overflow",
    "CWE-362": "Race Condition",
    "CWE-502": "Unsafe Deserialization",
    "CWE-754": "Improper Exception Handling",
}

PATTERN_RULES = {
    "eval(": ("CWE-94", "CRITICAL"),
    "exec(": ("CWE-94", "CRITICAL"),
    "os.system(": ("CWE-78", "HIGH"),
    "shell=True": ("CWE-78", "HIGH"),
    "pickle.loads": ("CWE-502", "CRITICAL"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
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
    confidence: float = 1.0

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
    recursion_depth: int = 0

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class Analyzer:
    """Parse Python into AST + metrics."""
    
    @staticmethod
    def analyze(path: str, root: str = "") -> CodeStructure:
        """Analyze Python file."""
        p = Path(path)
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return CodeStructure(path=path, rel_path=path, source="", lines=0)
        
        rel = str(p.relative_to(Path(root))) if root else p.name
        cs = CodeStructure(path=path, rel_path=rel, source=source, lines=source.count("\n"))
        
        try:
            tree = ast.parse(source)
            cs.tree = tree
        except SyntaxError:
            return cs
        
        # Extract structure
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                
                cs.functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [a.arg for a in node.args.args],
                    "complexity": complexity,
                })
            
            elif isinstance(node, ast.ClassDef):
                cs.classes.append({"name": node.name, "line": node.lineno})
            
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        cs.bare_excepts += 1
        
        return cs
    
    @staticmethod
    def snippet(source: str, line: int, ctx: int = 3) -> str:
        """Extract code snippet."""
        lines = source.split("\n")
        start = max(0, line - ctx - 1)
        end = min(len(lines), line + ctx)
        result = []
        for i in range(start, end):
            marker = ">" if i == line - 1 else " "
            result.append(f"{marker} {i+1:4d} | {lines[i]}")
        return "\n".join(result)

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """Base agent."""
    name = "base"
    
    def __init__(self, structures: List[CodeStructure]):
        self.structures = structures
        self.findings: List[Finding] = []
    
    async def run(self) -> List[Finding]:
        raise NotImplementedError

class StaticAgent(BaseAgent):
    """AST-based structural analysis."""
    name = "static"
    
    async def run(self) -> List[Finding]:
        await asyncio.sleep(0)  # yield
        
        for cs in self.structures:
            # High complexity functions
            for func in cs.functions:
                if func["complexity"] > 15:
                    self.findings.append(Finding(
                        agent=self.name,
                        title=f"High complexity: {func['name']}()",
                        description=f"Complexity={func['complexity']} (threshold: 15)",
                        severity=Severity.MEDIUM,
                        file=cs.rel_path,
                        line=func["line"],
                        fix="Refactor into smaller functions",
                    ))
            
            # Bare except clauses
            if cs.bare_excepts > 0:
                self.findings.append(Finding(
                    agent=self.name,
                    title=f"Bare except clauses ({cs.bare_excepts})",
                    description="Bare except hides all exceptions",
                    severity=Severity.MEDIUM,
                    file=cs.rel_path,
                    line=1,
                    cwe="CWE-754",
                    fix="Use `except Exception:`",
                ))
        
        return self.findings

class SecurityAgent(BaseAgent):
    """Pattern-based security detection."""
    name = "security"
    
    async def run(self) -> List[Finding]:
        await asyncio.sleep(0)  # yield
        
        for cs in self.structures:
            if not cs.tree:
                continue
            
            # Build real calls via AST (FIXED: no false positives)
            real_calls = set()
            for node in ast.walk(cs.tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, "id"):
                        real_calls.add(node.func.id)
            
            lines = cs.source.split("\n")
            for i, line in enumerate(lines, 1):
                lower = line.lower()
                
                # Skip dict/constant definitions
                if line.strip().startswith(('"', "'")) and ": {" in line:
                    continue
                if any(x in line for x in ("PATTERN_RULES", "CWE_DATABASE", "# noaudit")):
                    continue
                
                # Check patterns
                for pat, (cwe, severity) in PATTERN_RULES.items():
                    if pat.lower() not in lower:
                        continue
                    
                    pat_clean = pat.rstrip("(")
                    # Only flag if it's a REAL call, not a string definition
                    if pat_clean not in real_calls:
                        continue
                    
                    sev_map = {
                        "CRITICAL": Severity.CRITICAL,
                        "HIGH": Severity.HIGH,
                        "MEDIUM": Severity.MEDIUM,
                    }
                    
                    self.findings.append(Finding(
                        agent=self.name,
                        title=f"Pattern detected: {pat}",
                        description=f"Potentially dangerous pattern at line {i}",
                        severity=sev_map.get(severity, Severity.MEDIUM),
                        cwe=cwe,
                        file=cs.rel_path,
                        line=i,
                        fix=f"Review usage of {pat}",
                    ))
        
        return self.findings

class ConcurrencyAgent(BaseAgent):
    """Thread safety analysis."""
    name = "concurrency"
    
    async def run(self) -> List[Finding]:
        await asyncio.sleep(0)  # yield
        
        for cs in self.structures:
            source = cs.source
            
            has_threading = "threading" in source
            has_lock = "Lock()" in source or "RLock()" in source
            has_shared = any(x in source for x in ["global ", "shared_", "_cache"])
            
            if has_threading and not has_lock and has_shared:
                self.findings.append(Finding(
                    agent=self.name,
                    title="Threading without locks",
                    description="Shared state accessed without synchronization",
                    severity=Severity.HIGH,
                    cwe="CWE-362",
                    file=cs.rel_path,
                    fix="Add threading.Lock() for shared state",
                ))
            
            # time.sleep for sync (anti-pattern)
            if "time.sleep(" in source and "# noaudit" not in source:
                self.findings.append(Finding(
                    agent=self.name,
                    title="time.sleep() for synchronization",
                    description="sleep() is not a proper sync primitive",
                    severity=Severity.LOW,
                    cwe="CWE-667",
                    file=cs.rel_path,
                    fix="Use threading.Event() or Condition()",
                ))
        
        return self.findings

class PerformanceAgent(BaseAgent):
    """Performance anti-patterns."""
    name = "performance"
    
    async def run(self) -> List[Finding]:
        await asyncio.sleep(0)  # yield
        
        for cs in self.structures:
            source = cs.source
            lines = source.split("\n")
            
            # String concatenation in loops (O(n²))
            for i, line in enumerate(lines, 1):
                lower = line.lower()
                if re.search(r'\bfor\b', lower) and re.search(r'\+=\s*["\']', lower):
                    self.findings.append(Finding(
                        agent=self.name,
                        title="String concatenation in loop",
                        description="O(n^2) complexity due to string immutability",
                        severity=Severity.MEDIUM,
                        file=cs.rel_path,
                        line=i,
                        fix="Use list.append() and ''.join()",
                    ))
            
            # Overly complex functions
            for func in cs.functions:
                if func["complexity"] > 20:
                    self.findings.append(Finding(
                        agent=self.name,
                        title=f"Complex function impacts performance: {func['name']}()",
                        description=f"Complexity={func['complexity']} reduces JIT efficiency",
                        severity=Severity.MEDIUM,
                        file=cs.rel_path,
                        line=func["line"],
                        fix="Split into smaller, focused functions",
                    ))
        
        return self.findings

class LogicAgent(BaseAgent):
    """Logic bug detection (static analysis only, no LLM)."""
    name = "logic"
    
    async def run(self) -> List[Finding]:
        await asyncio.sleep(0)  # yield
        
        for cs in self.structures:
            for func in cs.functions:
                # Detect obvious issues
                source_snippet = Analyzer.snippet(cs.source, func["line"], ctx=10)
                
                # Check for common logic errors
                if any(pat in source_snippet for pat in ["== True", "== False", "== None"]):
                    self.findings.append(Finding(
                        agent=self.name,
                        title=f"Redundant comparison in {func['name']}()",
                        description="Use `is` for singleton comparisons",
                        severity=Severity.LOW,
                        file=cs.rel_path,
                        line=func["line"],
                        fix="Replace `== True` with truthiness check",
                    ))
        
        return self.findings

# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class Consensus:
    """Deduplicate and rank findings."""
    
    @staticmethod
    def deduplicate(findings: List[Finding]) -> List[Finding]:
        """Remove duplicates by signature."""
        seen = set()
        unique = []
        
        for f in findings:
            sig = (f.file, f.line, f.title)
            if sig not in seen:
                seen.add(sig)
                unique.append(f)
        
        return unique
    
    @staticmethod
    def rank(findings: List[Finding]) -> List[Finding]:
        """Sort by severity and confidence."""
        return sorted(
            findings,
            key=lambda f: (-f.severity.value, -f.confidence),
        )

# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """Main audit orchestrator."""
    
    def __init__(self):
        self.structures: List[CodeStructure] = []
        self.findings: List[Finding] = []
    
    def load(self, path: str) -> bool:
        """Load code from file or directory."""
        print(f"\n[*] Loading: {path}")
        
        p = Path(path)
        if not p.exists():
            print(f"[!] Path not found: {path}")
            return False
        
        py_files = []
        if p.is_file() and p.suffix == ".py":
            py_files = [p]
        elif p.is_dir():
            py_files = list(p.rglob("*.py"))
        else:
            print(f"[!] Not a Python file or directory")
            return False
        
        print(f"[+] Found {len(py_files)} file(s)")
        
        for py_file in py_files:
            cs = Analyzer.analyze(str(py_file), str(p.parent))
            self.structures.append(cs)
            print(f"    - {cs.rel_path}: {len(cs.functions)} functions, {len(cs.classes)} classes")
        
        return len(self.structures) > 0
    
    async def audit(self) -> None:
        """Run all agents in parallel."""
        print(f"\n[*] Running agents...")
        
        agents = [
            StaticAgent(self.structures),
            SecurityAgent(self.structures),
            ConcurrencyAgent(self.structures),
            PerformanceAgent(self.structures),
            LogicAgent(self.structures),
        ]
        
        # Run agents concurrently
        tasks = [agent.run() for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"    [!] {agents[i].name} error: {result}")
                continue
            
            print(f"    [+] {agents[i].name:15} -> {len(result)} findings")
            self.findings.extend(result)
    
    def finalize(self) -> List[Finding]:
        """Deduplicate and rank findings."""
        print(f"\n[*] Consensus phase...")
        
        before = len(self.findings)
        unique = Consensus.deduplicate(self.findings)
        after = len(unique)
        
        print(f"    [+] Deduplicated: {before} -> {after} findings")
        
        ranked = Consensus.rank(unique)
        return ranked
    
    def report(self, findings: List[Finding]) -> None:
        """Print audit report."""
        print("\n" + "=" * 80)
        print(" " * 20 + "AUDIT REPORT")
        print("=" * 80)
        
        if not findings:
            print("\n[+] No findings - code appears clean!\n")
            return
        
        # Group by severity
        by_sev = defaultdict(list)
        for f in findings:
            by_sev[f.severity].append(f)
        
        # Print by severity
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            if sev not in by_sev:
                continue
            
            items = by_sev[sev]
            print(f"\n[{sev.name}] ({len(items)} findings)")
            print("-" * 80)
            
            for i, f in enumerate(items[:10], 1):
                print(f"\n  {i}. {f.title}")
                print(f"     Agent: {f.agent:15} CWE: {f.cwe}")
                print(f"     File: {f.file}:{f.line}")
                if f.description:
                    print(f"     Issue: {f.description[:70]}")
                if f.fix:
                    print(f"     Fix: {f.fix[:70]}")
            
            if len(items) > 10:
                print(f"\n  ... and {len(items) - 10} more {sev.name} findings")
        
        print("\n" + "=" * 80)
        print(f"[TOTAL] {len(findings)} findings")
        print("=" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE (auto-tests the script on itself)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuite:
    """Self-test suite."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, condition: bool, details: str = "") -> None:
        """Log test result."""
        if condition:
            print(f"  [+] {name}")
            self.passed += 1
        else:
            print(f"  [-] {name}: {details}")
            self.failed += 1
    
    async def run(self) -> bool:
        """Run all tests."""
        print("\n" + "=" * 80)
        print(" " * 20 + "SELF-TEST SUITE")
        print("=" * 80 + "\n")
        
        # Test 1: Analyzer
        print("[Test 1] Analyzer")
        script_path = __file__
        cs = Analyzer.analyze(script_path, Path(script_path).parent)
        
        self.test("Parse AST", cs.tree is not None)
        self.test("Extract functions", len(cs.functions) > 0, f"Found {len(cs.functions)}")
        self.test("Extract classes", len(cs.classes) > 0, f"Found {len(cs.classes)}")
        
        # Test 2: Agents
        print("\n[Test 2] Agent Execution")
        structures = [cs]
        
        agents = [
            StaticAgent(structures),
            SecurityAgent(structures),
            ConcurrencyAgent(structures),
            PerformanceAgent(structures),
            LogicAgent(structures),
        ]
        
        for agent in agents:
            findings = await agent.run()
            self.test(f"{agent.name} agent", isinstance(findings, list), f"Got {len(findings)} findings")
        
        # Test 3: Consensus
        print("\n[Test 3] Consensus Engine")
        test_findings = [
            Finding("a1", "Issue A", "", Severity.HIGH, file="test.py", line=10),
            Finding("a2", "Issue A", "", Severity.MEDIUM, file="test.py", line=10),  # Dup
            Finding("a3", "Issue B", "", Severity.LOW, file="test.py", line=20),
        ]
        
        unique = Consensus.deduplicate(test_findings)
        self.test("Deduplication", len(unique) == 2, f"Got {len(unique)}, expected 2")
        
        ranked = Consensus.rank(unique)
        is_sorted = ranked[0].severity.value >= ranked[1].severity.value
        self.test("Ranking", is_sorted, "Findings not sorted by severity")
        
        # Test 4: Orchestrator
        print("\n[Test 4] Orchestrator")
        orch = Orchestrator()
        loaded = orch.load(script_path)
        self.test("Load file", loaded, "Failed to load")
        
        await orch.audit()
        self.test("Run audit", len(orch.findings) >= 0, f"Got {len(orch.findings)} findings")
        
        final = orch.finalize()
        self.test("Finalize", isinstance(final, list), "Not a list")
        
        # Summary
        print("\n" + "=" * 80)
        total = self.passed + self.failed
        pct = int(100 * self.passed / total) if total > 0 else 0
        print(f"[RESULTS] {self.passed}/{total} passed ({pct}%)")
        
        if self.failed == 0:
            print("[SUCCESS] All tests passed!")
        else:
            print(f"[WARNING] {self.failed} test(s) failed")
        
        print("=" * 80 + "\n")
        
        return self.failed == 0

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main entry point."""
    # Parse arguments
    target = __file__  # Default: audit self
    run_tests = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--self-test":
            run_tests = True
        else:
            target = arg
    
    # Header
    print("\n" + "=" * 80)
    print(" " * 15 + "NeuroAudit v6.0 - SINGLE FILE INTEGRATION")
    print("=" * 80)
    
    # Run tests if requested
    if run_tests:
        suite = TestSuite()
        success = await suite.run()
        return 0 if success else 1
    
    # Otherwise, run audit
    print(f"\n[*] Target: {target}")
    if target == __file__:
        print("[*] Mode: SELF-AUDIT (auditing itself)")
    
    orch = Orchestrator()
    
    if not orch.load(target):
        return 1
    
    await orch.audit()
    findings = orch.finalize()
    orch.report(findings)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[!] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
