"""Quick verify: alias_map with { } chars won't break str.format()."""
from strategy_forge.engine.graph_builder import _EXTRACT_PROMPT

# Simulate alias_map containing JSON with curly braces
base = _EXTRACT_PROMPT.format(
    text="__TEXT__",
    entity_types="Person, Organization",
    relation_types="ally, oppose",
    candidate_entities="美国, 苏联",
    alias_map='{"美国": ["USA"], "苏联": ["USSR"]}',
)
assert "美国" in base
assert "__TEXT__" in base
# Replace the placeholder
result = base.replace("__TEXT__", "特朗普访问北京后")
assert "特朗普访问北京" in result
assert "__TEXT__" not in result
# Verify no leftover { } from JSON are treated as format fields
# (if they were, .replace would have thrown KeyError)
print("VERIFIED: alias_map JSON passes through safely")
