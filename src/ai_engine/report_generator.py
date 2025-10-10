# src/ai_engine/report_generator.py — PRO++ ENHANCED
"""
Advanced HTML Report Generator with Jinja2 templating.

Features:
- Multi-path template resolution with fallbacks
- Thread-safe environment caching
- Comprehensive error handling with diagnostic HTML
- Custom Jinja2 filters and globals
- Environment variable configuration
- Path traversal protection
- Template hot-reloading support (dev mode)
- Rich fallback rendering
"""

from __future__ import annotations

import os
import json
import time
import logging
import pathlib
import traceback
import threading
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List, Literal, Callable
from functools import lru_cache

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
    TemplateNotFound,
    TemplateSyntaxError,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Logging
LOGGER = logging.getLogger(__name__)

# Default paths
THIS_FILE = pathlib.Path(__file__).resolve()
DEFAULT_ASSETS_DIR = THIS_FILE.parents[2] / "assets" / "templates"
DEFAULT_TEMPLATE = DEFAULT_ASSETS_DIR / "report_template.html"
FALLBACK_TEMPLATE_NAME = "report_template.html"

# Environment variables
ENV_TEMPLATES_DIR = os.getenv("REPORT_TEMPLATES_DIR")
ENV_DEV_MODE = os.getenv("REPORT_DEV_MODE", "false").lower() in ("true", "1", "yes")

# Limits
MAX_CONTEXT_JSON_SIZE = 1_000_000  # 1MB for JSON serialization
MAX_FALLBACK_CONTEXT_SIZE = 50_000  # 50KB for fallback display
CACHE_SIZE = 32  # LRU cache for _normalize_context

# =============================================================================
# THREAD-SAFE ENVIRONMENT CACHE
# =============================================================================

_env_cache: Dict[str, Environment] = {}
_env_lock = threading.Lock()


def _create_jinja_environment(templates_dir: pathlib.Path) -> Environment:
    """
    Create and configure Jinja2 Environment.
    
    Args:
        templates_dir: Directory containing templates
        
    Returns:
        Configured Environment instance
    """
    if not templates_dir.exists():
        LOGGER.warning(f"Templates directory does not exist: {templates_dir}")
    
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
        # Auto-reload in dev mode for hot-reloading
        auto_reload=ENV_DEV_MODE,
        # Cache size for compiled templates
        cache_size=400 if not ENV_DEV_MODE else 0,
    )
    
    # Register custom filters
    _register_custom_filters(env)
    
    # Register global functions
    _register_global_functions(env)
    
    LOGGER.info(
        f"Created Jinja2 environment: {templates_dir} "
        f"(auto_reload={ENV_DEV_MODE})"
    )
    
    return env


def _register_custom_filters(env: Environment) -> None:
    """Register custom Jinja2 filters."""
    
    # JSON formatting
    env.filters["tojson_pretty"] = lambda obj: json.dumps(
        obj, ensure_ascii=False, indent=2
    )
    env.filters["safe_json"] = lambda obj: json.dumps(
        obj, ensure_ascii=False
    )
    
    # Text formatting
    env.filters["nl2br"] = lambda s: str(s or "").replace("\n", "<br/>")
    env.filters["truncate_chars"] = lambda s, n=400: (
        s if len(str(s)) <= n else str(s)[:n - 1] + ""
    )
    
    # Number formatting
    env.filters["fmt_num"] = lambda x: (
        f"{x:,.0f}".replace(",", " ") if isinstance(x, (int, float)) else str(x)
    )
    env.filters["fmt_pct"] = lambda x: (
        f"{x:.2f}%" if isinstance(x, (int, float)) else str(x)
    )
    env.filters["fmt_float"] = lambda x, d=2: (
        f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)
    )
    
    # Date formatting
    env.filters["fmt_date"] = lambda s, fmt="%Y-%m-%d": (
        time.strftime(fmt, time.strptime(str(s), "%Y-%m-%d %H:%M:%S"))
        if s else ""
    )
    
    # Safety
    env.filters["safe_str"] = lambda s: str(s or "")
    env.filters["safe_list"] = lambda x: x if isinstance(x, list) else []
    env.filters["safe_dict"] = lambda x: x if isinstance(x, dict) else {}
    
    # HTML escaping
    env.filters["escape_html"] = _escape_html
    
    LOGGER.debug(f"Registered {len(env.filters)} custom filters")


def _register_global_functions(env: Environment) -> None:
    """Register global helper functions."""
    
    env.globals["now"] = lambda fmt="%Y-%m-%d %H:%M": time.strftime(fmt)
    env.globals["timestamp"] = lambda: int(time.time())
    env.globals["len"] = len
    env.globals["isinstance"] = isinstance
    env.globals["str"] = str
    
    LOGGER.debug(f"Registered {len(env.globals)} global functions")


def _get_or_create_environment(templates_dir: pathlib.Path) -> Environment:
    """
    Get cached Environment or create new one (thread-safe).
    
    Args:
        templates_dir: Templates directory
        
    Returns:
        Jinja2 Environment
    """
    key = str(templates_dir.resolve())
    
    with _env_lock:
        if key not in _env_cache:
            _env_cache[key] = _create_jinja_environment(templates_dir)
        return _env_cache[key]


def clear_environment_cache() -> None:
    """Clear all cached Jinja2 environments."""
    with _env_lock:
        _env_cache.clear()
    LOGGER.info("Cleared Jinja2 environment cache")


# =============================================================================
# TEMPLATE RESOLUTION
# =============================================================================

def _get_candidate_template_dirs(
    template_path: Optional[pathlib.Path]
) -> List[pathlib.Path]:
    """
    Get list of directories to search for templates (in priority order).
    
    Priority:
        1. Directory containing template_path (if provided)
        2. REPORT_TEMPLATES_DIR environment variable
        3. ./assets/templates (current working directory)
        4. DEFAULT_ASSETS_DIR (package default)
    
    Args:
        template_path: Optional specific template path
        
    Returns:
        List of candidate directories (deduplicated)
    """
    candidates: List[pathlib.Path] = []
    
    # 1. Explicit template path directory
    if template_path:
        candidates.append(template_path.parent.resolve())
    
    # 2. Environment variable
    if ENV_TEMPLATES_DIR:
        env_path = pathlib.Path(ENV_TEMPLATES_DIR).resolve()
        if env_path.exists():
            candidates.append(env_path)
        else:
            LOGGER.warning(
                f"REPORT_TEMPLATES_DIR points to non-existent path: {env_path}"
            )
    
    # 3. CWD assets/templates
    cwd_assets = (pathlib.Path.cwd() / "assets" / "templates").resolve()
    candidates.append(cwd_assets)
    
    # 4. Package default
    candidates.append(DEFAULT_ASSETS_DIR.resolve())
    
    # Deduplicate while preserving order
    seen = set()
    unique: List[pathlib.Path] = []
    for d in candidates:
        key = str(d)
        if key not in seen:
            unique.append(d)
            seen.add(key)
    
    LOGGER.debug(f"Template search paths: {[str(p) for p in unique]}")
    
    return unique


def _validate_template_path(path: pathlib.Path) -> bool:
    """
    Validate template path for security (prevent path traversal).
    
    Args:
        path: Path to validate
        
    Returns:
        True if path is safe
    """
    try:
        # Resolve to absolute path
        resolved = path.resolve()
        
        # Check if it's within one of the allowed directories
        for allowed_dir in _get_candidate_template_dirs(None):
            try:
                resolved.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        
        LOGGER.warning(f"Template path outside allowed directories: {path}")
        return False
        
    except Exception as e:
        LOGGER.error(f"Error validating template path: {e}")
        return False


def _locate_template(
    template_path: Optional[pathlib.Path]
) -> tuple[pathlib.Path, pathlib.Path, bool]:
    """
    Locate template file in candidate directories.
    
    Args:
        template_path: Optional specific template path
        
    Returns:
        Tuple of (loader_dir, template_file, exists)
    """
    # If specific path provided and exists, use it
    if template_path:
        if not _validate_template_path(template_path):
            LOGGER.error(f"Invalid template path: {template_path}")
        elif template_path.exists():
            LOGGER.info(f"Using explicit template: {template_path}")
            return template_path.parent, template_path, True
    
    # Determine filename to search for
    filename = (
        template_path.name if template_path 
        else FALLBACK_TEMPLATE_NAME
    )
    
    # Search in candidate directories
    for directory in _get_candidate_template_dirs(template_path):
        candidate = directory / filename
        if candidate.exists():
            LOGGER.info(f"Found template: {candidate}")
            return directory, candidate, True
    
    # Not found - return last directory as fallback
    dirs = _get_candidate_template_dirs(template_path)
    fallback_dir = dirs[-1] if dirs else DEFAULT_ASSETS_DIR
    fallback_path = fallback_dir / filename
    
    LOGGER.warning(f"Template not found: {filename}")
    
    return fallback_dir, fallback_path, False


# =============================================================================
# CONTEXT NORMALIZATION
# =============================================================================

# Default context values
_DEFAULT_CONTEXT: Dict[str, Any] = {
    "title": "Business Report",
    "subtitle": "",
    "run_meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M")},
    "metrics": {},
    "notes": "",
    "kpis": None,
    "tags": None,
    "data_dictionary": None,
    "forecast": None,
    "anomalies": None,
    "recommendations": None,
    "insights": None,
    "model_card": None,
    "df_preview": None,
}


@lru_cache(maxsize=CACHE_SIZE)
def _get_default_context() -> Dict[str, Any]:
    """Get default context (cached)."""
    return _DEFAULT_CONTEXT.copy()


def _normalize_context(context: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Normalize and validate context dictionary.
    
    Args:
        context: Input context (may be None or incomplete)
        
    Returns:
        Complete context with defaults
    """
    if context is None:
        context = {}
    
    if not isinstance(context, dict):
        LOGGER.error(f"Context must be dict, got {type(context)}")
        context = {}
    
    # Merge with defaults
    merged = {**_get_default_context(), **context}
    
    # Ensure run_meta has timestamp
    if "run_meta" not in merged or not isinstance(merged["run_meta"], dict):
        merged["run_meta"] = {}
    
    if "timestamp" not in merged["run_meta"]:
        merged["run_meta"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
    
    # Validate JSON serializability (for debug)
    try:
        json_str = json.dumps(merged, ensure_ascii=False)
        if len(json_str) > MAX_CONTEXT_JSON_SIZE:
            LOGGER.warning(
                f"Context JSON size exceeds {MAX_CONTEXT_JSON_SIZE} bytes: "
                f"{len(json_str)}"
            )
    except (TypeError, ValueError) as e:
        LOGGER.warning(f"Context contains non-serializable data: {e}")
    
    return merged


def _truncate_context_for_fallback(context: dict[str, Any]) -> dict[str, Any]:
    """
    Truncate large context for fallback HTML display.
    
    Args:
        context: Full context
        
    Returns:
        Truncated context
    """
    truncated = {}
    
    for key, value in context.items():
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            if len(json_str) > 1000:  # Truncate large values
                truncated[key] = "<truncated: {len(json_str)} chars>"
            else:
                truncated[key] = value
        except Exception:
            truncated[key] = str(type(value))
    
    return truncated


# =============================================================================
# RENDERING
# =============================================================================

def build_report_html(
    context: dict[str, Any],
    template_path: str | pathlib.Path | None = None
) -> str:
    """
    Render HTML report using Jinja2 template.
    
    This is the main public API - maintains backward compatibility.
    
    Args:
        context: Report context data (may be incomplete)
        template_path: Optional path to custom template
        
    Returns:
        Rendered HTML (always returns valid HTML, even on error)
    """
    try:
        # Convert string path to Path object
        tpl_path = pathlib.Path(template_path) if template_path else None
        
        # Locate template
        loader_dir, template_file, exists = _locate_template(tpl_path)
        
        # If template doesn't exist, return fallback
        if not exists:
            LOGGER.error(f"Template not found: {template_file}")
            return _render_missing_template_fallback(
                template_file, loader_dir, context
            )
        
        # Get or create Jinja2 environment
        env = _get_or_create_environment(loader_dir)
        
        # Load template
        try:
            template = env.get_template(template_file.name)
        except TemplateNotFound:
            LOGGER.error(f"Jinja2 TemplateNotFound: {template_file.name}")
            return _render_missing_template_fallback(
                template_file, loader_dir, context
            )
        except TemplateSyntaxError as e:
            LOGGER.error(f"Template syntax error: {e}")
            return _render_template_syntax_error_fallback(
                e, template_file, loader_dir, context
            )
        
        # Normalize context
        normalized_context = _normalize_context(context)
        
        # Render template
        html = template.render(**normalized_context)
        
        LOGGER.info(
            f"Successfully rendered report: {len(html)} bytes "
            f"(template: {template_file.name})"
        )
        
        return html
        
    except Exception as e:
        LOGGER.exception("Unexpected error rendering report")
        return _render_exception_fallback(
            e, context, loader_dir if 'loader_dir' in locals() else None,
            template_file if 'template_file' in locals() else None
        )


# =============================================================================
# DIAGNOSTICS API
# =============================================================================

@dataclass(frozen=True)
class RenderDiagnostics:
    """Diagnostic information about template rendering."""
    ok: bool
    message: str
    loader_dir: str
    template_file: str
    template_exists: bool
    candidates: List[str]
    context_keys: List[str]
    context_size_bytes: int
    render_time_ms: float
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RenderResult:
    """Result of template rendering with diagnostics."""
    html: str
    diagnostics: RenderDiagnostics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "html": self.html,
            "diagnostics": self.diagnostics.to_dict()
        }


def render_report_html_pro(
    context: dict[str, Any],
    template_path: str | pathlib.Path | None = None
) -> RenderResult:
    """
    Render report with comprehensive diagnostics.
    
    Useful for testing, debugging, and CI/CD.
    
    Args:
        context: Report context
        template_path: Optional custom template
        
    Returns:
        RenderResult with HTML and diagnostics
    """
    start_time = time.time()
    
    try:
        # Convert path
        tpl_path = pathlib.Path(template_path) if template_path else None
        
        # Locate template
        loader_dir, template_file, exists = _locate_template(tpl_path)
        
        # Get candidates
        candidates = [
            str(p) for p in _get_candidate_template_dirs(tpl_path)
        ]
        
        # Context info
        context_keys = list(context.keys()) if context else []
        try:
            context_json = json.dumps(context or {}, ensure_ascii=False)
            context_size = len(context_json.encode('utf-8'))
        except Exception:
            context_size = 0
        
        # If template missing
        if not exists:
            html = _render_missing_template_fallback(
                template_file, loader_dir, context
            )
            
            render_time = (time.time() - start_time) * 1000
            
            diagnostics = RenderDiagnostics(
                ok=False,
                message="Template file not found",
                loader_dir=str(loader_dir),
                template_file=str(template_file),
                template_exists=False,
                candidates=candidates,
                context_keys=context_keys,
                context_size_bytes=context_size,
                render_time_ms=render_time,
                traceback=None
            )
            
            return RenderResult(html=html, diagnostics=diagnostics)
        
        # Render template
        env = _get_or_create_environment(loader_dir)
        template = env.get_template(template_file.name)
        normalized_context = _normalize_context(context)
        html = template.render(**normalized_context)
        
        render_time = (time.time() - start_time) * 1000
        
        diagnostics = RenderDiagnostics(
            ok=True,
            message="OK",
            loader_dir=str(loader_dir),
            template_file=str(template_file),
            template_exists=True,
            candidates=candidates,
            context_keys=context_keys,
            context_size_bytes=context_size,
            render_time_ms=render_time,
            traceback=None
        )
        
        LOGGER.info(
            f"Report rendered successfully: {len(html)} bytes in "
            f"{render_time:.2f}ms"
        )
        
        return RenderResult(html=html, diagnostics=diagnostics)
        
    except Exception as e:
        LOGGER.exception("Error in render_report_html_pro")
        
        tb = traceback.format_exc()
        render_time = (time.time() - start_time) * 1000
        
        html = _render_exception_fallback(
            e, context,
            loader_dir if 'loader_dir' in locals() else None,
            template_file if 'template_file' in locals() else None
        )
        
        diagnostics = RenderDiagnostics(
            ok=False,
            message=str(e),
            loader_dir=str(loader_dir) if 'loader_dir' in locals() else "unknown",
            template_file=str(template_file) if 'template_file' in locals() else "unknown",
            template_exists=exists if 'exists' in locals() else False,
            candidates=candidates if 'candidates' in locals() else [],
            context_keys=context_keys if 'context_keys' in locals() else [],
            context_size_bytes=context_size if 'context_size' in locals() else 0,
            render_time_ms=render_time,
            traceback=tb
        )
        
        return RenderResult(html=html, diagnostics=diagnostics)


# =============================================================================
# FALLBACK HTML RENDERERS
# =============================================================================

def _render_missing_template_fallback(
    template_path: pathlib.Path,
    loader_dir: pathlib.Path,
    context: dict[str, Any]
) -> str:
    """Render fallback HTML when template is missing."""
    normalized = _normalize_context(context)
    truncated = _truncate_context_for_fallback(normalized)
    
    candidates = _get_candidate_template_dirs(None)
    candidates_html = "\n".join([
        "<li><code>{_escape_html(str(p))}</code> "
        "{'✅' if p.exists() else '❌'}</li>"
        for p in candidates
    ])
    
    return """<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Report Template Missing</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #d32f2f; margin-top: 0; }}
        h2 {{ color: #333; margin-top: 30px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 6px;
            overflow: auto;
            border-left: 4px solid #2196F3;
        }}
        .info {{ background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        .warning {{ background: #fff3e0; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        ul {{ line-height: 1.8; }}
        small {{ color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Report Template Not Found</h1>
        
        <div class="warning">
            <strong>Missing file:</strong> <code>{_escape_html(str(template_path))}</code>
        </div>
        
        <h2>📂 Search Paths (in order)</h2>
        <ul>{candidates_html}</ul>
        
        <div class="info">
            <strong>💡 Solution:</strong> Place your template file in one of the directories above,
            or set the <code>REPORT_TEMPLATES_DIR</code> environment variable.
        </div>
        
        <h2>📊 Minimal Data Preview</h2>
        <p><strong>{_escape_html(normalized.get('title', 'Report'))}</strong></p>
        {f"<p><small>{_escape_html(normalized.get('subtitle', ''))}</small></p>" if normalized.get('subtitle') else ""}
        
        <h3>Context Keys</h3>
        <pre>{_escape_html(_safe_json(list(truncated.keys())))}</pre>
        
        <h3>Context Data (truncated)</h3>
        <pre>{_escape_html(_safe_json(truncated))}</pre>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        <small>
            Loader directory: <code>{_escape_html(str(loader_dir))}</code><br>
            Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
        </small>
    </div>
</body>
</html>"""


def _render_template_syntax_error_fallback(
    error: TemplateSyntaxError,
    template_path: pathlib.Path,
    loader_dir: pathlib.Path,
    context: dict[str, Any]
) -> str:
    """Render fallback HTML for template syntax errors."""
    return """<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Template Syntax Error</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #d32f2f; margin-top: 0; }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 6px;
            overflow: auto;
            border-left: 4px solid #f44336;
        }}
        .error {{ background: #ffebee; padding: 15px; border-radius: 6px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Template Syntax Error</h1>
        
        <div class="error">
            <strong>Error:</strong> {_escape_html(str(error))}<br>
            <strong>Line:</strong> {error.lineno if hasattr(error, 'lineno') else 'unknown'}
        </div>
        
        <p><strong>Template:</strong> <code>{_escape_html(str(template_path))}</code></p>
        <p><strong>Loader:</strong> <code>{_escape_html(str(loader_dir))}</code></p>
        
        <h2>💡 Common Issues</h2>
        <ul>
            <li>Unclosed tags: <code>{% if %}</code> without <code>{% endif %}</code></li>
            <li>Invalid variable syntax: check {{ }} and {% %}</li>
            <li>Missing filters or undefined variables</li>
        </ul>
        
        <small>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
</body>
</html>"""


def _render_exception_fallback(
    error: Exception,
    context: dict[str, Any],
    loader_dir: Optional[pathlib.Path],
    template_path: Optional[pathlib.Path]
) -> str:
    """Render fallback HTML for unexpected exceptions."""
    tb = traceback.format_exc()
    normalized = _normalize_context(context)
    truncated = _truncate_context_for_fallback(normalized)
    
    return """<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Report Rendering Error</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #d32f2f; margin-top: 0; }}
        h2 {{ color: #333; margin-top: 25px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 6px;
            overflow: auto;
            border-left: 4px solid #f44336;
            font-size: 0.85em;
        }}
        .error {{ background: #ffebee; padding: 15px; border-radius: 6px; margin: 20px 0; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Report Rendering Error</h1>
        
        <div class="error">
            <strong>Error Type:</strong> {_escape_html(type(error).__name__)}<br>
            <strong>Message:</strong> {_escape_html(str(error))}
        </div>
        
        <h2>📍 Diagnostics</h2>
        <ul>
            <li><strong>Loader Directory:</strong> <code>{_escape_html(str(loader_dir or 'unknown'))}</code></li>
            <li><strong>Template File:</strong> <code>{_escape_html(str(template_path or 'unknown'))}</code></li>
            <li><strong>Context Keys:</strong> {len(context) if context else0}</li>
        </ul>
        
        <h2>🔍 Full Traceback</h2>
        <pre>{_escape_html(tb)}</pre>
        
        <h2>📊 Context Preview (truncated)</h2>
        <pre>{_escape_html(_safe_json(truncated))}</pre>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        <small>
            Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}<br>
            Report Generator Version: PRO++ ENHANCED
        </small>
    </div>
</body>
</html>"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _escape_html(text: str) -> str:
    """
    Escape HTML special characters.
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-safe text
    """
    if not isinstance(text, str):
        text = str(text)
    
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _safe_json(obj: Any) -> str:
    """
    Safely convert object to JSON string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string or error message
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except TypeError as e:
        LOGGER.warning(f"JSON serialization failed: {e}")
        return "<non-serializable: {type(obj).__name__}>"
    except Exception as e:
        LOGGER.error(f"Unexpected JSON error: {e}")
        return str(obj)


# =============================================================================
# TESTING & DEBUG UTILITIES
# =============================================================================

def validate_template(
    template_path: str | pathlib.Path
) -> tuple[bool, str]:
    """
    Validate template file for syntax errors.
    
    Args:
        template_path: Path to template
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        tpl_path = pathlib.Path(template_path)
        
        if not tpl_path.exists():
            return False, f"Template file does not exist: {tpl_path}"
        
        loader_dir = tpl_path.parent
        env = _get_or_create_environment(loader_dir)
        
        # Try to get template (this will check syntax)
        env.get_template(tpl_path.name)
        
        return True, "Template is valid"
        
    except TemplateSyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.message}"
    except Exception as e:
        return False, f"Validation error: {e}"


def get_template_info(
    template_path: Optional[str | pathlib.Path] = None
) -> Dict[str, Any]:
    """
    Get information about template resolution.
    
    Useful for debugging template loading issues.
    
    Args:
        template_path: Optional template path
        
    Returns:
        Dictionary with template information
    """
    tpl_path = pathlib.Path(template_path) if template_path else None
    loader_dir, template_file, exists = _locate_template(tpl_path)
    
    candidates = _get_candidate_template_dirs(tpl_path)
    
    return {
        "template_path": str(template_path or "default"),
        "resolved_template": str(template_file),
        "template_exists": exists,
        "loader_directory": str(loader_dir),
        "search_paths": [str(p) for p in candidates],
        "search_paths_exist": [p.exists() for p in candidates],
        "env_templates_dir": ENV_TEMPLATES_DIR,
        "dev_mode": ENV_DEV_MODE,
        "cached_environments": len(_env_cache),
    }


def render_test_report(
    include_all_features: bool = True
) -> str:
    """
    Render a test report with sample data.
    
    Useful for testing template rendering.
    
    Args:
        include_all_features: Include all optional features
        
    Returns:
        Rendered HTML
    """
    test_context = {
        "title": "Test Report - PRO++",
        "subtitle": "Template rendering validation",
        "run_meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
            "environment": "test"
        },
        "metrics": {
            "test_metric_1": 42,
            "test_metric_2": 3.14159,
            "test_metric_3": "success"
        },
        "notes": "This is a test report.\nIt contains multiple lines.\nUsed for validation.",
    }
    
    if include_all_features:
        test_context.update({
            "kpis": [
                {"label": "Total Records", "value": "1,000", "status": "ok"},
                {"label": "Missing Data", "value": "5.2%", "status": "warn"},
                {"label": "Duplicates", "value": "0", "status": "ok"},
            ],
            "tags": ["test", "validation", "pro++"],
            "data_dictionary": {
                "columns": ["id", "name", "value"],
                "rows": [
                    {"column": "id", "dtype": "int64", "missing_pct": 0.0},
                    {"column": "name", "dtype": "object", "missing_pct": 2.1},
                    {"column": "value", "dtype": "float64", "missing_pct": 5.2},
                ]
            },
            "insights": [
                "Test insight number 1",
                "Test insight number 2",
                "Test insight number 3"
            ],
            "recommendations": [
                "Test recommendation 1",
                "Test recommendation 2"
            ],
        })
    
    return build_report_html(test_context)


# =============================================================================
# PUBLIC API SUMMARY
# =============================================================================

__all__ = [
    # Main rendering functions
    "build_report_html",
    "render_report_html_pro",
    
    # Data classes
    "RenderResult",
    "RenderDiagnostics",
    
    # Utility functions
    "validate_template",
    "get_template_info",
    "render_test_report",
    "clear_environment_cache",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _init_module():
    """Initialize module on import."""
    # Log configuration
    LOGGER.info(f"Report Generator initialized")
    LOGGER.info(f"Default templates dir: {DEFAULT_ASSETS_DIR}")
    LOGGER.info(f"Dev mode: {ENV_DEV_MODE}")
    
    if ENV_TEMPLATES_DIR:
        LOGGER.info(f"Custom templates dir: {ENV_TEMPLATES_DIR}")
    
    # Validate default template exists (warn if missing)
    if not DEFAULT_TEMPLATE.exists():
        LOGGER.warning(
            f"Default template not found: {DEFAULT_TEMPLATE}\n"
            f"Reports will use fallback HTML."
        )


# Initialize on import
_init_module()