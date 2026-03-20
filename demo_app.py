"""Demo application for cjm-transcript-segment-align library.

Demonstrates the dual-column Segmentation & Alignment step with shared chrome,
zone switching, and cross-domain alignment status. Works standalone without
the full transcript workflow.

Run with: python demo_app.py
"""

from typing import List, Dict, Any, Callable
from pathlib import Path
import tempfile

from fasthtml.common import (
    fast_app, Div, H1, P, Span, Input, Button,
    APIRouter, FileResponse,
)

# DaisyUI components
from cjm_fasthtml_daisyui.core.resources import get_daisyui_headers
from cjm_fasthtml_daisyui.core.testing import create_theme_persistence_script
from cjm_fasthtml_daisyui.components.data_display.badge import badge, badge_styles, badge_sizes
from cjm_fasthtml_daisyui.utilities.semantic_colors import bg_dui, text_dui, border_dui
from cjm_fasthtml_daisyui.utilities.border_radius import border_radius

# Tailwind utilities
from cjm_fasthtml_tailwind.utilities.spacing import p, m
from cjm_fasthtml_tailwind.utilities.sizing import w, h, min_h, container, max_w
from cjm_fasthtml_tailwind.utilities.typography import font_size, font_weight, uppercase, tracking
from cjm_fasthtml_tailwind.utilities.layout import overflow, display_tw
from cjm_fasthtml_tailwind.utilities.borders import border
from cjm_fasthtml_tailwind.utilities.effects import ring
from cjm_fasthtml_tailwind.utilities.transitions_and_animation import transition, duration
from cjm_fasthtml_tailwind.utilities.flexbox_and_grid import (
    flex_display, flex_direction, justify, items, gap, grow
)
from cjm_fasthtml_tailwind.core.base import combine_classes

# App core
from cjm_fasthtml_app_core.core.routing import register_routes
from cjm_fasthtml_app_core.core.htmx import handle_htmx_request

# Interactions library
from cjm_fasthtml_interactions.core.state_store import get_session_id

# State store
from cjm_workflow_state.state_store import SQLiteWorkflowStateStore

# Plugin system
from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.scheduling import SafetyScheduler

# Card stack library
from cjm_fasthtml_card_stack.components.states import render_loading_state
from cjm_fasthtml_card_stack.core.constants import DEFAULT_VISIBLE_COUNT, DEFAULT_CARD_WIDTH

# Segmentation library
from cjm_transcript_segmentation.models import SegmentationUrls
from cjm_transcript_segmentation.services.segmentation import SegmentationService
from cjm_transcript_segmentation.html_ids import SegmentationHtmlIds
from cjm_transcript_segmentation.components.card_stack_config import SEG_CS_IDS
from cjm_transcript_segmentation.routes.init import init_segmentation_routers

# Alignment library
from cjm_transcript_vad_align.models import AlignmentUrls
from cjm_transcript_vad_align.services.alignment import AlignmentService
from cjm_transcript_vad_align.html_ids import AlignmentHtmlIds
from cjm_transcript_vad_align.components.card_stack_config import ALIGN_CS_IDS
from cjm_transcript_vad_align.routes.init import init_alignment_routers

# Combined library (this library)
from cjm_transcript_segment_align.html_ids import CombinedHtmlIds
from cjm_transcript_segment_align.components.handlers import (
    wrapped_seg_split, wrapped_seg_merge, wrapped_seg_undo,
    wrapped_seg_reset, wrapped_seg_ai_split,
    create_seg_init_chrome_wrapper, create_align_init_chrome_wrapper,
)
from cjm_transcript_segment_align.components.step_renderer import (
    render_alignment_status,
)
from cjm_transcript_segment_align.components.keyboard_config import SWITCH_CHROME_BTN_ID
from cjm_transcript_segment_align.routes.chrome import init_chrome_router
from cjm_transcript_segment_align.routes.forced_alignment import (
    init_forced_alignment_routers, FA_CONTAINER_ID, render_fa_controls,
)
from cjm_transcript_segment_align.services.forced_alignment import ForcedAlignmentService


# =============================================================================
# Test Audio Files
# =============================================================================

TEST_FILES_DIR = Path(__file__).parent / "test_files"
TEST_AUDIO_PATH = TEST_FILES_DIR / "short_test_audio.mp3"

SAMPLE_TEXT = (TEST_FILES_DIR / "short_test_audio.txt").read_text().strip() if (TEST_FILES_DIR / "short_test_audio.txt").exists() else "Sample text for segmentation demo."


# =============================================================================
# Mock Services
# =============================================================================

class MockSourceService:
    """Mock source service that provides both text and audio path mapping."""

    def __init__(self, text_blocks: List[Dict], path_map: Dict[str, str]):
        self._text_blocks = text_blocks
        self._path_map = path_map

    def get_source_blocks(self, selected_sources: List[Dict]) -> List[Any]:
        """Return sample text as source blocks."""
        from cjm_source_provider.models import SourceBlock
        return [
            SourceBlock(
                id=src["record_id"],
                provider_id=src["provider_id"],
                text=SAMPLE_TEXT,
            )
            for src in selected_sources
        ]

    def get_transcription_by_id(self, record_id: str, provider_id: str) -> Any:
        """Return mock source block with media path."""
        from dataclasses import dataclass

        @dataclass
        class MockBlock:
            media_path: str

        path = self._path_map.get(record_id, "")
        return MockBlock(media_path=path)


# =============================================================================
# Demo Page Renderer
# =============================================================================

def render_demo_page(
    seg_urls: SegmentationUrls,
    align_urls: AlignmentUrls,
    switch_chrome_url: str,
) -> Callable:
    """Create the demo page content factory."""

    def page_content():
        """Render the demo page with dual-column layout."""

        # --- Segmentation column ---
        seg_header = Div(
            Span(
                "Text Segmentation",
                cls=combine_classes(
                    font_size.sm, font_weight.bold,
                    uppercase, tracking.wide,
                    text_dui.base_content.opacity(50)
                )
            ),
            Span(
                "--",
                id=CombinedHtmlIds.SEG_MINI_STATS,
                cls=combine_classes(badge, badge_styles.ghost, badge_sizes.sm)
            ),
            id=CombinedHtmlIds.SEG_COLUMN_HEADER,
            cls=combine_classes(
                flex_display, justify.between, items.center,
                p(3), bg_dui.base_200,
                border_dui.base_300, border.b()
            )
        )

        seg_content = Div(
            render_loading_state(SEG_CS_IDS, message="Initializing segments..."),
            Div(
                hx_post=seg_urls.init,
                hx_trigger="load",
                hx_target=f"#{CombinedHtmlIds.SEG_COLUMN_CONTENT}",
                hx_swap="outerHTML"
            ),
            id=CombinedHtmlIds.SEG_COLUMN_CONTENT,
            cls=combine_classes(grow(), overflow.hidden, flex_display, flex_direction.col, p(4))
        )

        seg_column_cls = combine_classes(
            w.full, w('[60%]').lg,
            min_h(0),
            flex_display, flex_direction.col,
            bg_dui.base_100, border_dui.base_300, border(1),
            border_radius.box,
            overflow.hidden,
            transition.all, duration._200,
            ring(1), "ring-primary",
        )

        seg_col = Div(seg_header, seg_content, id=CombinedHtmlIds.SEG_COLUMN, cls=seg_column_cls)

        # --- Alignment column ---
        align_header = Div(
            Span(
                "VAD Alignment",
                cls=combine_classes(
                    font_size.sm, font_weight.bold,
                    uppercase, tracking.wide,
                    text_dui.base_content.opacity(50)
                )
            ),
            Span(
                "--",
                id=CombinedHtmlIds.ALIGNMENT_MINI_STATS,
                cls=combine_classes(badge, badge_styles.ghost, badge_sizes.sm)
            ),
            id=CombinedHtmlIds.ALIGNMENT_COLUMN_HEADER,
            cls=combine_classes(
                flex_display, justify.between, items.center,
                p(3), bg_dui.base_200,
                border_dui.base_300, border.b()
            )
        )

        align_content = Div(
            render_loading_state(ALIGN_CS_IDS, message="Analyzing audio..."),
            Div(
                hx_post=align_urls.init,
                hx_trigger="load",
                hx_target=f"#{CombinedHtmlIds.ALIGNMENT_COLUMN_CONTENT}",
                hx_swap="outerHTML"
            ),
            id=CombinedHtmlIds.ALIGNMENT_COLUMN_CONTENT,
            cls=combine_classes(grow(), min_h(0), overflow.hidden, flex_display, flex_direction.col, p(4))
        )

        align_column_cls = combine_classes(
            w.full, w('[40%]').lg,
            min_h(0),
            flex_display, flex_direction.col,
            bg_dui.base_100, border_dui.base_300, border(1),
            border_radius.box,
            overflow.hidden,
            transition.all, duration._200,
            "opacity-70",
        )

        align_col = Div(align_header, align_content, id=CombinedHtmlIds.ALIGNMENT_COLUMN, cls=align_column_cls)

        # --- Placeholder chrome ---
        placeholder_cls = combine_classes(font_size.sm, text_dui.base_content.opacity(50))

        hints = Div(
            P("Keyboard hints will appear here after initialization.", cls=placeholder_cls),
            id=CombinedHtmlIds.SHARED_HINTS,
            cls=str(p(2))
        )

        toolbar = Div(
            P("Toolbar actions will appear here after initialization.", cls=placeholder_cls),
            id=CombinedHtmlIds.SHARED_TOOLBAR,
            cls=str(p(2))
        )

        # FA controls container (empty until seg init populates it)
        fa_controls = Div(id=FA_CONTAINER_ID, cls=combine_classes(flex_display, items.center, gap(2)))

        controls = Div(
            P("Width controls will appear here after initialization.", cls=placeholder_cls),
            id=CombinedHtmlIds.SHARED_CONTROLS,
            cls=str(p(2))
        )

        footer = Div(
            P("Footer will appear here after initialization.", cls=placeholder_cls),
            id=CombinedHtmlIds.SHARED_FOOTER,
            cls=combine_classes(
                p(1), bg_dui.base_100,
                border_dui.base_300, border.t(),
                flex_display, justify.center, items.center
            )
        )

        # Hidden active column input + KB system container
        active_column_input = Input(
            type="hidden",
            id=CombinedHtmlIds.ACTIVE_COLUMN_INPUT,
            name="active_column",
            value="seg",
        )
        kb_container = Div(id=CombinedHtmlIds.KEYBOARD_SYSTEM)

        # Hidden chrome switch button placeholder (populated by seg init OOB swap)
        chrome_switch_btn = Button(
            id=SWITCH_CHROME_BTN_ID,
            cls=str(display_tw.hidden),
            hx_post=switch_chrome_url,
            hx_include=f"#{CombinedHtmlIds.ACTIVE_COLUMN_INPUT}",
            hx_swap="none",
        )

        return Div(
            # Header
            Div(
                H1("Segment & Align Demo",
                   cls=combine_classes(font_size._3xl, font_weight.bold)),
                P(
                    "Dual-column text segmentation and VAD alignment with shared chrome and zone switching.",
                    cls=combine_classes(text_dui.base_content.opacity(70), m.b(2))
                ),
            ),

            # Shared chrome
            hints,
            toolbar,
            fa_controls,
            controls,

            # Dual-column content area
            Div(
                seg_col,
                align_col,
                cls=combine_classes(
                    grow(),
                    min_h(0),
                    flex_display,
                    flex_direction.col,
                    flex_direction.row.lg,
                    gap(4),
                    overflow.hidden,
                    p(1),
                )
            ),

            # Footer
            footer,

            # Keyboard system container
            kb_container,

            # Hidden state + chrome switch button
            active_column_input,
            chrome_switch_btn,

            id="sa-demo-container",
            cls=combine_classes(
                w.full, h.full,
                flex_display, flex_direction.col,
                p(4), p.x(2), p.b(0)
            )
        )

    return page_content


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
        hdrs=[*get_daisyui_headers(), create_theme_persistence_script()],
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
    plugin_manager = PluginManager(scheduler=SafetyScheduler())
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

    # -------------------------------------------------------------------------
    # Create services
    # -------------------------------------------------------------------------
    # Map demo source to test audio file
    selected_sources = [
        {"record_id": "demo-source-0", "provider_id": "demo-provider"},
    ]
    path_map = {"demo-source-0": str(TEST_AUDIO_PATH)}

    source_service = MockSourceService(
        text_blocks=[{"id": "demo-source-0", "text": SAMPLE_TEXT}],
        path_map=path_map,
    )
    segmentation_service = SegmentationService(plugin_manager, nltk_plugin_name)
    alignment_service = AlignmentService(plugin_manager, vad_plugin_name)

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

    fa_service = ForcedAlignmentService(plugin_manager, fa_plugin_name)
    fa_is_available = fa_service.is_available()

    # Initialize selection state
    def init_demo_state(sess):
        """Ensure demo state is initialized for session."""
        session_id = get_session_id(sess)
        workflow_state = state_store.get_state(workflow_id, session_id)
        if "step_states" not in workflow_state:
            workflow_state["step_states"] = {}
        if "selection" not in workflow_state["step_states"]:
            workflow_state["step_states"]["selection"] = {
                "selected_sources": selected_sources
            }
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
    # Set up segmentation routes (with wrapped mutation handlers)
    # -------------------------------------------------------------------------
    wrapped_handlers = {
        "split": wrapped_seg_split,
        "merge": wrapped_seg_merge,
        "undo": wrapped_seg_undo,
        "reset": wrapped_seg_reset,
        "ai_split": wrapped_seg_ai_split,
    }

    seg_routers, seg_urls, seg_routes = init_segmentation_routers(
        state_store=state_store,
        workflow_id=workflow_id,
        source_service=source_service,
        segmentation_service=segmentation_service,
        prefix="/seg",
        wrapped_handlers=wrapped_handlers,
    )

    # -------------------------------------------------------------------------
    # Set up alignment routes
    # -------------------------------------------------------------------------
    align_routers, align_urls, align_routes = init_alignment_routers(
        state_store=state_store,
        workflow_id=workflow_id,
        source_service=source_service,
        alignment_service=alignment_service,
        prefix="/align",
        audio_src_url=audio_src_url,
    )

    # -------------------------------------------------------------------------
    # Set up chrome switching route (from this library)
    # -------------------------------------------------------------------------
    chrome_router, chrome_routes = init_chrome_router(
        state_store=state_store,
        workflow_id=workflow_id,
        seg_urls=seg_urls,
        align_urls=align_urls,
        prefix="/chrome",
    )
    switch_chrome_url = chrome_routes["switch_chrome"].to()

    # -------------------------------------------------------------------------
    # Set up forced alignment routes
    # -------------------------------------------------------------------------
    fa_router, fa_routes = init_forced_alignment_routers(
        state_store=state_store,
        workflow_id=workflow_id,
        fa_service=fa_service,
        source_service=source_service,
        seg_urls=seg_urls,
        prefix="/fa",
    )
    fa_trigger_url = fa_routes["trigger"].to() if fa_is_available else ""
    fa_toggle_url = fa_routes["toggle"].to() if fa_is_available else ""

    # -------------------------------------------------------------------------
    # Override init routes with combined wrappers
    # -------------------------------------------------------------------------
    # Seg init wrapper (builds combined KB system + shared chrome + FA controls)
    wrapped_seg_init_fn = create_seg_init_chrome_wrapper(
        align_urls=align_urls,
        switch_chrome_url=switch_chrome_url,
        fa_trigger_url=fa_trigger_url,
        fa_toggle_url=fa_toggle_url,
        fa_available=fa_is_available,
    )

    seg_init_router = APIRouter(prefix="/seg/workflow")

    @seg_init_router
    async def init(request, sess):
        """Initialize segments with combined KB system."""
        init_demo_state(sess)
        return await wrapped_seg_init_fn(
            state_store, workflow_id, source_service, segmentation_service,
            request, sess, urls=seg_urls,
        )

    # Align init wrapper (adds mini-stats + alignment status)
    wrapped_align_init_fn = create_align_init_chrome_wrapper()

    align_init_router = APIRouter(prefix="/align/workflow")

    @align_init_router
    async def init(request, sess):
        """Initialize alignment with mini-stats and status."""
        init_demo_state(sess)
        return await wrapped_align_init_fn(
            state_store, workflow_id, source_service, alignment_service,
            request, sess, urls=align_urls,
        )

    # -------------------------------------------------------------------------
    # Page routes
    # -------------------------------------------------------------------------
    page_content = render_demo_page(seg_urls, align_urls, switch_chrome_url)

    @router
    def index(request, sess):
        """Demo homepage."""
        init_demo_state(sess)
        return handle_htmx_request(request, page_content)

    # -------------------------------------------------------------------------
    # Register routes
    # -------------------------------------------------------------------------
    register_routes(
        app, router, audio_router, chrome_router, fa_router,
        seg_init_router, align_init_router,
        *seg_routers, *align_routers,
    )

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
