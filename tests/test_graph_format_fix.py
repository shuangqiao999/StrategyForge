"""Verify Template migration: alias_map JSON with { } passes through safely."""
from string import Template
from strategy_forge.engine.graph_builder import _EXTRACT_PROMPT

t = Template(_EXTRACT_PROMPT)
base = t.substitute(
    text="__TEXT__",
    entity_types="Person, Organization",
    relation_types="ally, oppose",
    candidate_entities="美国, 苏联",
    alias_map='{"美国": ["USA"], "苏联": ["USSR"]}',
)
assert "美国" in base
assert "__TEXT__" in base
result = base.replace("__TEXT__", "特朗普访问北京后")
assert "特朗普" in result
assert "__TEXT__" not in result
print("VERIFIED: Template safely passes JSON braces")
