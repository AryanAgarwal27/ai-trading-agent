"""Tools the orchestrator uses to talk to Freqtrade and to the world.

Per BRD §1.1 rule 2, only Freqtrade touches the exchange. The modules in this
package are the orchestrator's side of the boundary: REST client, subprocess
driver, regime bucketing, paper-container lifecycle. None of them place orders
directly.
"""

from orchestrator.tools.freqtrade_lifecycle import (
    PaperLifecycleError,
    PaperSpawnError,
    PaperSpawnTimeout,
    PaperStopError,
    next_free_paper_port,
    spawn_paper_container,
    stop_paper_container,
)

__all__ = [
    "PaperLifecycleError",
    "PaperSpawnError",
    "PaperSpawnTimeout",
    "PaperStopError",
    "next_free_paper_port",
    "spawn_paper_container",
    "stop_paper_container",
]
