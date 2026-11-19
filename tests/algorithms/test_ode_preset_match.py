"""P2-2: ODE preset 匹配不再依赖子串巧合（supply↛supply_chain，resource↛energy_resource）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy_forge.algorithms.module_utils import _match_ode_preset


ODE_MAP = {
    "fatigue": "fatigue_recovery", "supply": "supply_consumption",
    "pollution": "pollution_spread", "resources": "resource_depletion",
    "population": "logistic", "economy": "logistic",
    "market_share": "logistic", "cash_flow": "decay", "brand": "logistic",
}


def test_exact_match():
    assert _match_ode_preset("fatigue", ODE_MAP) == "fatigue_recovery"
    assert _match_ode_preset("supply", ODE_MAP) == "supply_consumption"
    assert _match_ode_preset("market_share", ODE_MAP) == "logistic"


def test_supply_chain_not_matched_to_supply():
    assert _match_ode_preset("supply_chain", ODE_MAP) is None


def test_energy_resource_not_matched_to_resource():
    assert _match_ode_preset("energy_resource", ODE_MAP) is None


def test_no_match_returns_none():
    assert _match_ode_preset("totally_unknown", ODE_MAP) is None


def test_prefix_match_still_works_for_safe_patterns():
    assert _match_ode_preset("market_share_index", ODE_MAP) == "logistic"
