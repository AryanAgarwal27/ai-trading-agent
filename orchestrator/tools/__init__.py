"""Tools the orchestrator uses to talk to Freqtrade and to the world.

Per BRD §1.1 rule 2, only Freqtrade touches the exchange. The modules in this
package are the orchestrator's side of the boundary: REST client, subprocess
driver, regime bucketing. None of them place orders directly.
"""
