"""
Opik Tracing & Observability Integration for FoodLoversTV GraphRAG.
Supports automated tracing of:
- Knowledge Graph hybrid retrieval & subgraph extraction
- Gemini LLM generation, model fallback cascade, and prompts
- End-to-end user query execution & conversational threads
- Pipeline scraping and entity enrichment batch jobs

Includes graceful fallback: if the opik package is not installed or
credentials are not supplied, all tracing decorators act as seamless pass-throughs.
"""

import os
import sys
import logging
import functools
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opik_tracer")

# Detect whether opik is available
try:
    import opik
    from opik import track as _native_opik_track
    from opik import opik_context
    OPIK_AVAILABLE = True
except ImportError:
    opik = None
    _native_opik_track = None
    opik_context = None
    OPIK_AVAILABLE = False

DEFAULT_PROJECT_NAME = "FoodLoversTV-GraphRAG"

# Runtime configuration state
_CONFIG = {
    "api_key": None,
    "workspace": None,
    "project_name": DEFAULT_PROJECT_NAME,
    "enabled": True,
    "initialized": False,
    "use_local": False,
}


def load_opik_key(key_file: str = "opik_api.txt") -> Optional[str]:
    """Retrieve Opik API key from Streamlit secrets, environment variables, or local key file."""
    # 1. Try Streamlit secrets
    try:
        import streamlit as st
        if "OPIK_API_KEY" in st.secrets:
            return str(st.secrets["OPIK_API_KEY"]).strip()
    except Exception:
        pass

    # 2. Try Environment Variable
    if os.environ.get("OPIK_API_KEY"):
        return os.environ["OPIK_API_KEY"].strip()

    # 3. Try Local Key File
    candidates = [key_file, "opik_key.txt", "../opik_api.txt"]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                    if val:
                        return val
            except Exception as e:
                logger.warning(f"Failed to read Opik key from {path}: {e}")

    return None


def configure_opik(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    use_local: bool = False,
    enabled: bool = True,
) -> bool:
    """
    Configure Opik client globally. Returns True if Opik is active, False otherwise.
    """
    global _CONFIG

    _CONFIG["enabled"] = enabled
    if not enabled:
        return False

    # Check for local mode flag
    if use_local or os.environ.get("OPIK_USE_LOCAL", "").lower() in ("1", "true"):
        _CONFIG["use_local"] = True

    resolved_api_key = api_key or load_opik_key()
    resolved_workspace = (
        workspace
        or os.environ.get("OPIK_WORKSPACE")
        or (_get_streamlit_secret("OPIK_WORKSPACE") if "streamlit" in sys.modules else None)
    )
    resolved_project = (
        project_name
        or os.environ.get("OPIK_PROJECT_NAME")
        or (_get_streamlit_secret("OPIK_PROJECT_NAME") if "streamlit" in sys.modules else None)
        or DEFAULT_PROJECT_NAME
    )

    _CONFIG["api_key"] = resolved_api_key
    _CONFIG["workspace"] = resolved_workspace
    _CONFIG["project_name"] = resolved_project

    # Set standard environment variables for underlying SDK
    if resolved_api_key:
        os.environ["OPIK_API_KEY"] = resolved_api_key
    if resolved_workspace:
        os.environ["OPIK_WORKSPACE"] = resolved_workspace
    if resolved_project:
        os.environ["OPIK_PROJECT_NAME"] = resolved_project

    if not OPIK_AVAILABLE:
        logger.info("opik package not installed. Tracing is currently running in pass-through mode.")
        _CONFIG["initialized"] = False
        return False

    try:
        if _CONFIG["use_local"]:
            opik.configure(use_local=True, project_name=resolved_project)
            logger.info(f"Opik configured in LOCAL mode for project '{resolved_project}'.")
        elif resolved_api_key:
            opik.configure(
                api_key=resolved_api_key,
                workspace=resolved_workspace,
                project_name=resolved_project,
            )
            logger.info(f"Opik configured for workspace '{resolved_workspace}', project '{resolved_project}'.")
        else:
            # Opik may already be configured via ~/.opik.config or interactive login
            opik.configure(project_name=resolved_project)
            logger.info(f"Opik configured with existing credentials for project '{resolved_project}'.")

        _CONFIG["initialized"] = True
        return True

    except Exception as e:
        logger.warning(f"Could not initialize Opik client: {e}. Tracing will be non-blocking.")
        _CONFIG["initialized"] = False
        return False


def _get_streamlit_secret(key: str) -> Optional[str]:
    """Helper to safely fetch a secret from Streamlit."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def is_opik_active() -> bool:
    """Check if Opik is currently installed and actively tracing."""
    return bool(OPIK_AVAILABLE and _CONFIG["enabled"] and _CONFIG["initialized"])


def get_opik_status() -> Dict[str, Any]:
    """Return dictionary of Opik operational status for UI diagnostics."""
    return {
        "installed": OPIK_AVAILABLE,
        "active": is_opik_active(),
        "enabled": _CONFIG["enabled"],
        "has_api_key": bool(_CONFIG["api_key"]),
        "workspace": _CONFIG.get("workspace"),
        "project_name": _CONFIG.get("project_name", DEFAULT_PROJECT_NAME),
        "use_local": _CONFIG.get("use_local", False),
    }


def track(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project_name: Optional[str] = None,
    **kwargs,
):
    """
    Safe decorator that wraps functions with Opik's @track decorator if Opik is available and active.
    If Opik is not installed or unconfigured, it gracefully passes through with zero overhead.

    Can be used as:
        @track
        def my_func(): ...

    or with arguments:
        @track(name="My Step", tags=["graphrag"])
        def my_func(): ...
    """
    def decorator(target_fn: Callable) -> Callable:
        if OPIK_AVAILABLE and _native_opik_track is not None and _CONFIG["enabled"]:
            try:
                track_args = {**kwargs}
                if name:
                    track_args["name"] = name
                if type:
                    track_args["type"] = type
                if tags:
                    track_args["tags"] = tags
                if metadata:
                    track_args["metadata"] = metadata
                if project_name or _CONFIG.get("project_name"):
                    track_args["project_name"] = project_name or _CONFIG.get("project_name")

                # Wrap with native Opik track decorator
                return _native_opik_track(**track_args)(target_fn)
            except Exception as e:
                logger.debug(f"Failed to apply opik.track to {target_fn.__name__}: {e}")

        # Fallback pass-through wrapper
        @functools.wraps(target_fn)
        def wrapper(*args, **call_kwargs):
            return target_fn(*args, **call_kwargs)
        return wrapper

    if func is not None and callable(func):
        return decorator(func)
    return decorator


def update_current_trace(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    **kwargs,
):
    """Safely update active Opik trace properties mid-execution."""
    if not is_opik_active() or opik_context is None:
        return

    try:
        update_args = {**kwargs}
        if name:
            update_args["name"] = name
        if tags:
            update_args["tags"] = tags
        if metadata:
            update_args["metadata"] = metadata
        if thread_id:
            update_args["thread_id"] = thread_id

        opik_context.update_current_trace(**update_args)
    except Exception as e:
        logger.debug(f"Opik update_current_trace failed: {e}")


def update_current_span(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
    **kwargs,
):
    """Safely update active Opik span properties (such as LLM tokens, model info, metadata)."""
    if not is_opik_active() or opik_context is None:
        return

    try:
        update_args = {**kwargs}
        if name:
            update_args["name"] = name
        if tags:
            update_args["tags"] = tags
        if metadata:
            update_args["metadata"] = metadata
        if model:
            update_args["model"] = model
        if provider:
            update_args["provider"] = provider
        if usage:
            update_args["usage"] = usage

        opik_context.update_current_span(**update_args)
    except Exception as e:
        logger.debug(f"Opik update_current_span failed: {e}")


def get_current_trace_id() -> Optional[str]:
    """Retrieve the active Opik trace ID if currently within a trace."""
    if not is_opik_active() or opik_context is None:
        return None

    try:
        trace_data = opik_context.get_current_trace_data()
        if trace_data and hasattr(trace_data, "id"):
            return str(trace_data.id)
    except Exception as e:
        logger.debug(f"Could not get trace ID: {e}")
    return None


def get_trace_url(trace_id: Optional[str] = None) -> Optional[str]:
    """Generate the Opik dashboard link for a given trace ID."""
    tid = trace_id or get_current_trace_id()
    if not tid:
        return None

    if _CONFIG.get("use_local"):
        return f"http://localhost:5173/default/projects/{_CONFIG['project_name']}/traces/{tid}"

    workspace = _CONFIG.get("workspace")
    if workspace:
        return f"https://www.comet.com/opik/{workspace}/projects/{_CONFIG['project_name']}/traces/{tid}"
    return f"https://www.comet.com/opik"


# Auto-configure on import if keys or environment variables exist
try:
    configure_opik()
except Exception as _e:
    logger.debug(f"Auto-configuration of Opik failed on import: {_e}")
