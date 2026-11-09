"""Prospective analytical sensitivity scenarios; never empirical model evidence.

Paired accuracy normal approximation: n=(z_alpha+z_power)^2*(q-delta^2)/delta^2.
q is assumed discordance, delta is assumed net improvement. Macro-F1/joint
power and small-sample exact McNemar power still require development data.
"""
import json
import math
from statistics import NormalDist

normal = NormalDist()
scenarios = []
for family, alpha in (("primary_two_endpoints_Holm_bound", 0.025), ("secondary_four_tests_Holm_bound", 0.0125)):
    for delta in (0.03, 0.05, 0.08):
        for q in (0.10, 0.20, 0.30):
            n = math.ceil((normal.inv_cdf(1-alpha/2) + normal.inv_cdf(0.90))**2 * (q-delta**2) / delta**2)
            scenarios.append({"family": family, "alpha_two_sided": alpha, "marginal_power_assumed": 0.90,
                              "assumed_accuracy_delta": delta, "assumed_discordance": q,
                              "n_independent_claims_approx": n, "n_with_assumed_design_effect_1_2": math.ceil(1.2*n)})
print(json.dumps({"status": "PLANNING_ONLY_NOT_OBSERVED_POWER", "macro_f1_power": None,
                  "joint_power": None, "assumptions": "Normal approximation; 90% marginal target for eventual >=80% joint via union bound, conditional on adequate power for BOTH endpoints. Cluster factor 1.2 is hypothetical.",
                  "scenarios": scenarios}, indent=2))
