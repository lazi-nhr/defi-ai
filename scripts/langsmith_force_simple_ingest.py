from __future__ import annotations

import os

def force_simple_langsmith_ingest() -> None:
    """
    Force LangSmith to avoid multipart ingestion endpoints.

    Some workspaces/accounts do not permit /runs/multipart (403),
    while allowing standard ingestion. This function sets the
    runtime flags used by the LangSmith client to prefer simple ingest.
    """
    # Keep your existing env settings, but ensure they exist at runtime.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # These are the known knobs across langsmith versions.
    # We set multiple to cover client differences.
    os.environ["LANGSMITH_DISABLE_MULTIPART"] = "true"
    os.environ["LANGSMITH_MULTIPART_DISABLED"] = "true"
    os.environ["LANGSMITH_USE_MULTIPART"] = "false"
    os.environ["LANGSMITH_RUNS_MULTIPART_ENABLED"] = "false"
    os.environ["LANGSMITH_GZIP"] = "false"
    os.environ["LANGSMITH_COMPRESS"] = "false"
