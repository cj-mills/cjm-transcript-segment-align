"""Demo application for cjm-transcript-segment-align library.

Demonstrates the dual-column Segmentation & Alignment step with shared chrome,
zone switching, and cross-domain alignment status. Works standalone without
the full transcript workflow.

Run with: python demo_app.py
"""

from typing import List, Dict, Any
from pathlib import Path
import tempfile

from fasthtml.common import fast_app, Div, APIRouter, FileResponse

# DaisyUI components
from cjm_fasthtml_daisyui.core.resources import get_daisyui_headers
from cjm_fasthtml_daisyui.core.testing import create_theme_persistence_script

# App core
from cjm_fasthtml_app_core.core.routing import register_routes
from cjm_fasthtml_app_core.core.htmx import handle_htmx_request

# Interactions library
from cjm_fasthtml_interactions.core.context import InteractionContext
from cjm_fasthtml_interactions.core.state_store import get_session_id

# State store
from cjm_workflow_state.state_store import SQLiteWorkflowStateStore

# Plugin system
from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.scheduling import QueueScheduler
from cjm_plugin_system.core.queue import JobQueue

# SSE headers (always included — harmless if FA unavailable)
from cjm_fasthtml_job_monitor.components.modal import get_sse_headers

# Combined library (this library) — consolidated API
from cjm_transcript_segment_align.routes.init import init_segment_align_routers


# =============================================================================
# Test Audio Files — two sources for multi-source FA testing
# =============================================================================

TEST_FILES_DIR = Path(__file__).parent / "test_files"

TEST_SOURCES = [
    {
        "record_id": "demo-source-0",
        "audio": TEST_FILES_DIR / "short_test_audio.mp3",
        "text": (TEST_FILES_DIR / "short_test_audio.txt").read_text().strip(),
    },
    {
        "record_id": "demo-source-1",
        "audio": TEST_FILES_DIR / "02 - 1. Laying Plans.mp3",
        "text": (TEST_FILES_DIR / "02 - 1. Laying Plans.txt").read_text().strip(),
    },
]

TEST_AUDIO_PATH = TEST_SOURCES[0]["audio"]  # First source for audio serving fallback


# =============================================================================
# Mock Services
# =============================================================================

class MockSourceService:
    """Mock source service that provides both text and audio path mapping."""

    def __init__(self, source_map: Dict[str, Dict]):
        self._source_map = source_map  # record_id -> {"text": ..., "audio": ...}

    def get_source_blocks(self, selected_sources: List[Dict]) -> List[Any]:
        """Return source blocks with text from the source map."""
        from cjm_source_provider.models import SourceBlock
        return [
            SourceBlock(
                id=src["record_id"],
                provider_id=src["provider_id"],
                text=self._source_map[src["record_id"]]["text"],
            )
            for src in selected_sources
        ]

    def get_transcription_by_id(self, record_id: str, provider_id: str) -> Any:
        """Return mock source block with media path."""
        from dataclasses import dataclass

        @dataclass
        class MockBlock:
            media_path: str

        info = self._source_map.get(record_id, {})
        return MockBlock(media_path=str(info.get("audio", "")))


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Initialize the combined demo and start the server."""
    print("\n" + "=" * 70)
    print("Initializing cjm-transcript-segment-align Demo")
    print("=" * 70)

    # Initialize FastHTML app
    app, rt = fast_app(
        pico=False,
        hdrs=[*get_daisyui_headers(), create_theme_persistence_script(), *get_sse_headers()],
        title="Segment & Align Demo",
        htmlkw={'data-theme': 'light'},
        secret_key="demo-secret-key"
    )

    router = APIRouter(prefix="")

    # -------------------------------------------------------------------------
    # Set up state store
    # -------------------------------------------------------------------------
    temp_db = Path(tempfile.gettempdir()) / "cjm_transcript_segment_align_demo_state.db"
    # Clean state on each launch so demo always starts fresh
    if temp_db.exists():
        temp_db.unlink()
    state_store = SQLiteWorkflowStateStore(temp_db)
    workflow_id = "sa-demo"

    print(f"  State store: {temp_db} (fresh)")

    # -------------------------------------------------------------------------
    # Set up plugin manager and load plugins
    # -------------------------------------------------------------------------
    print("\n[Plugin System]")
    plugin_manager = PluginManager(scheduler=QueueScheduler())
    plugin_manager.discover_manifests()

    # Load NLTK plugin (segmentation)
    nltk_plugin_name = "cjm-text-plugin-nltk"
    nltk_meta = plugin_manager.get_discovered_meta(nltk_plugin_name)
    if nltk_meta:
        try:
            success = plugin_manager.load_plugin(nltk_meta, {"language": "english"})
            print(f"  {nltk_plugin_name}: {'loaded' if success else 'failed'}")
        except Exception as e:
            print(f"  {nltk_plugin_name}: error - {e}")
    else:
        print(f"  {nltk_plugin_name}: not found")

    # Load Silero VAD plugin (alignment)
    vad_plugin_name = "cjm-media-plugin-silero-vad"
    vad_meta = plugin_manager.get_discovered_meta(vad_plugin_name)
    if vad_meta:
        try:
            success = plugin_manager.load_plugin(vad_meta, {"threshold": 0.5})
            print(f"  {vad_plugin_name}: {'loaded' if success else 'failed'}")
        except Exception as e:
            print(f"  {vad_plugin_name}: error - {e}")
    else:
        print(f"  {vad_plugin_name}: not found")

    # Load Qwen3 forced alignment plugin (optional)
    fa_plugin_name = "cjm-transcription-plugin-qwen3-forced-aligner"
    fa_meta = plugin_manager.get_discovered_meta(fa_plugin_name)
    if fa_meta:
        try:
            success = plugin_manager.load_plugin(fa_meta, {"language": "English"})
            print(f"  {fa_plugin_name}: {'loaded' if success else 'failed'}")
        except Exception as e:
            print(f"  {fa_plugin_name}: error - {e}")
    else:
        print(f"  {fa_plugin_name}: not found (FA button will be hidden)")

    # System monitor (optional, for GPU stats in Resources tab)
    sysmon_name = "cjm-system-monitor-nvidia"
    sysmon_meta = plugin_manager.get_discovered_meta(sysmon_name)
    sysmon_available = False
    if sysmon_meta:
        try:
            success = plugin_manager.load_plugin(sysmon_meta)
            sysmon_available = success
            print(f"  {sysmon_name}: {'loaded' if success else 'failed'}")
        except Exception as e:
            print(f"  {sysmon_name}: error - {e}")
    else:
        print(f"  {sysmon_name}: not found (Resources tab will show CPU/RAM only)")

    # -------------------------------------------------------------------------
    # Create mock source service — multi-source
    # -------------------------------------------------------------------------
    selected_sources = [
        {"record_id": src["record_id"], "provider_id": "demo-provider"}
        for src in TEST_SOURCES
    ]
    source_map = {
        src["record_id"]: {"text": src["text"], "audio": src["audio"]}
        for src in TEST_SOURCES
    }
    source_service = MockSourceService(source_map=source_map)

    # -------------------------------------------------------------------------
    # Job queue (host-owned)
    # -------------------------------------------------------------------------
    queue = JobQueue(plugin_manager)

    # Initialize selection state + decomposition step state for job monitor
    def init_demo_state(sess):
        """Ensure demo state is initialized for session."""
        session_id = get_session_id(sess)
        workflow_state = state_store.get_state(workflow_id, session_id)
        if "step_states" not in workflow_state:
            workflow_state["step_states"] = {}
        changed = False
        if "selection" not in workflow_state["step_states"]:
            workflow_state["step_states"]["selection"] = {
                "selected_sources": selected_sources
            }
            changed = True
        if "decomposition" not in workflow_state["step_states"]:
            workflow_state["step_states"]["decomposition"] = {}
            changed = True
        if changed:
            state_store.update_state(workflow_id, session_id, workflow_state)

    # -------------------------------------------------------------------------
    # Audio serving route
    # -------------------------------------------------------------------------
    audio_router = APIRouter(prefix="/audio")

    @audio_router
    def audio_src(path: str = None):
        """Serve audio file for Web Audio API playback."""
        if path and Path(path).exists():
            return FileResponse(path, media_type="audio/mpeg")
        if TEST_AUDIO_PATH.exists():
            return FileResponse(str(TEST_AUDIO_PATH), media_type="audio/mpeg")
        from fasthtml.common import Response
        return Response(status_code=404, content="Audio file not found")

    audio_src_url = audio_src.to()

    # -------------------------------------------------------------------------
    # Initialize all segment-align routes (consolidated)
    # -------------------------------------------------------------------------
    sa_routers, sa_result = init_segment_align_routers(
        state_store=state_store,
        workflow_id=workflow_id,
        prefix="",
        source_service=source_service,
        plugin_manager=plugin_manager,
        job_queue=queue,
        audio_src_url=audio_src_url,
        text_plugin=nltk_plugin_name,
        vad_plugin=vad_plugin_name,
        fa_plugin_name=fa_plugin_name,
        sysmon_plugin_name=sysmon_name if sysmon_available else None,
        max_history_depth=10,
    )

    print(f"\n  FA available: {sa_result.fa_available}")

    # -------------------------------------------------------------------------
    # Page route — uses render_combined_step via sa_result.render_step
    # -------------------------------------------------------------------------
    @router
    def index(request, sess):
        """Demo homepage."""
        init_demo_state(sess)
        session_id = get_session_id(sess)
        workflow_state = state_store.get_state(workflow_id, session_id)
        ctx = InteractionContext(state=workflow_state, session=sess)
        return handle_htmx_request(request, sa_result.render_step(ctx))

    # -------------------------------------------------------------------------
    # Register routes
    # -------------------------------------------------------------------------
    register_routes(app, router, audio_router, *sa_routers)

    # -------------------------------------------------------------------------
    # Job queue lifecycle
    # -------------------------------------------------------------------------
    @app.on_event("startup")
    async def on_startup():
        await queue.start()
        print("Job queue started")

    @app.on_event("shutdown")
    async def on_shutdown():
        await queue.stop()
        plugin_manager.unload_all()
        print("Job queue stopped, plugins unloaded")

    # Debug output
    print("\n" + "=" * 70)
    print("Registered Routes:")
    print("=" * 70)
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  {route.path}")
    print("=" * 70)
    print("Demo App Ready!")
    print("=" * 70 + "\n")

    return app


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    app = main()

    port = 5036
    host = "0.0.0.0"
    display_host = 'localhost' if host in ['0.0.0.0', '127.0.0.1'] else host

    print(f"Server: http://{display_host}:{port}")
    print()
    print("Controls:")
    print("  Arrow Up/Down       - Navigate segments/chunks")
    print("  Arrow Left/Right    - Switch between columns")
    print("  Enter/Space         - Enter split mode (segmentation)")
    print("  Escape              - Exit split mode")
    print("  Backspace           - Merge with previous segment")
    print("  Ctrl+Z              - Undo")
    print("  [ / ]               - Adjust viewport width")
    print()

    timer = threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}"))
    timer.daemon = True
    timer.start()

    uvicorn.run(app, host=host, port=port)
