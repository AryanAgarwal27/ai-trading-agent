"""ai-trading-agent orchestrator package. See BRD.md and SPEC.md for design.

Stage 11c (F1): load ``.env`` HERE, at package import, so it runs before any
orchestrator submodule pulls in LangGraph. LangGraph reads
``LANGGRAPH_STRICT_MSGPACK`` at IMPORT time
(``langgraph/checkpoint/serde/_msgpack.py``) to decide whether to restrict
checkpoint deserialization to a safe type set (BRD §6.6/§15 — blocks code
execution from a compromised checkpoint DB). Importing ``orchestrator.main`` (or
any submodule) imports this package first, so loading ``.env`` here guarantees
the flag is in ``os.environ`` before that capture. ``main.py`` previously called
``load_dotenv()`` only AFTER importing langgraph, which left strict mode silently
OFF when the flag was supplied via ``.env`` (the documented method). ``override``
is left at its default (False) so a real process-environment value always wins.
"""

from dotenv import load_dotenv

load_dotenv()

__version__ = "0.0.1"
