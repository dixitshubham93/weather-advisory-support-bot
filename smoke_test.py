import sys
sys.path.insert(0, 'backend')

from policy_engine import load_sops
sops = load_sops()
categories = set(s["category"] for s in sops)
severities = set(s["severity"] for s in sops)
fuzzy_sops = [s for s in sops if any(c.get("operator") == "fuzzy" for c in s.get("conditions", []))]

print(f"SOPs loaded: {len(sops)}")
print(f"Categories ({len(categories)}): {sorted(categories)}")
print(f"Severities ({len(severities)}): {sorted(severities)}")
print(f"Fuzzy SOPs: {[s['id'] for s in fuzzy_sops]}")

# Now compile the graph (requires Ollama running locally -- skipping LLM calls here)
try:
    from graph import graph
    print("LangGraph compiled OK")
    print("Graph nodes:", list(graph.nodes.keys()) if hasattr(graph, 'nodes') else "compiled")
except Exception as e:
    print(f"Graph compile note: {e}")
