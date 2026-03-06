from __future__ import annotations

import argparse
import itertools
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
from PIL import Image, ImageOps

try:
    from pigbet_slice_views import VIEW_SPECS, build_middle_view_triptych
except ModuleNotFoundError:
    from inference.pigbet_slice_views import VIEW_SPECS, build_middle_view_triptych

try:
    from PySide6.QtCore import QEvent, QTimer, Qt, Signal
    from PySide6.QtGui import QCloseEvent, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PySide6 is required for the orientation helper. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
APP_TITLE = "PigBET Orientation Helper"
CARD_PANEL_SIZE = (210, 210)
REFERENCE_PANEL_SIZE = (320, 320)
CARD_SPACING = 18
CARD_WIDGET_WIDTH = CARD_PANEL_SIZE[0] * len(VIEW_SPECS) + (CARD_SPACING * (len(VIEW_SPECS) - 1)) + 64
BROWSER_MIN_COLUMNS = 1
BROWSER_COLUMN_GAP = 24
SIDEBAR_WIDTH = 430
BUNDLED_REFERENCE_VOLUME = REPO_ROOT / "Example_images-selected" / "Pig_1.nii.gz"
BUNDLED_REFERENCE_PREVIEW = MODULE_DIR / "assets" / "reference_mid_triptych.png"
COLORS = {
    "bg": "#f4efe6",
    "panel": "#fbf8f2",
    "surface": "#ffffff",
    "surface_soft": "#f6f2e9",
    "border": "#d9d1c5",
    "text": "#19232b",
    "muted": "#5a6770",
    "accent": "#0f766e",
    "accent_hover": "#115e59",
    "selected": "#c2410c",
}


@dataclass(frozen=True)
class OrientationSpec:
    axes: tuple[str, str, str]
    label: str
    slug: str
    is_identity: bool


@dataclass
class CandidateResult:
    spec: OrientationSpec
    volume_path: Path
    preview_path: Path
    left_right_flip: bool = False
    warning_text: str = ""


def strip_nifti_extension(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def ensure_nii_gz_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.name.endswith(".nii.gz"):
        return raw
    if raw.suffix == ".nii":
        return raw.with_suffix(".nii.gz")
    return raw.with_name(f"{strip_nifti_extension(raw.name)}.nii.gz")


def derive_default_output_path(input_path: str | Path) -> Path:
    source = Path(input_path).expanduser()
    base = strip_nifti_extension(source.name)
    return source.with_name(f"{base}_oriented.nii.gz")


def build_orientation_specs() -> list[OrientationSpec]:
    specs: list[OrientationSpec] = []
    for permutation in itertools.permutations(("x", "y", "z")):
        for flips in itertools.product((False, True), repeat=3):
            axes = tuple(f"-{axis}" if flip else axis for axis, flip in zip(permutation, flips))
            label = " ".join(axes)
            slug = "_".join(axis.replace("-", "neg") for axis in axes)
            specs.append(
                OrientationSpec(
                    axes=axes,
                    label=label,
                    slug=slug,
                    is_identity=axes == ("x", "y", "z"),
                )
            )

    specs.sort(key=lambda item: (not item.is_identity, item.label.replace("-", "~")))
    return specs


ORIENTATION_SPECS = build_orientation_specs()


def resolve_fslswapdim_path(explicit_path: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    resolved = shutil.which("fslswapdim")
    if resolved:
        candidates.append(Path(resolved))

    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidates.append(Path(fsldir).expanduser() / "bin" / "fslswapdim")

    candidates.extend(
        [
            Path.home() / "fsl" / "bin" / "fslswapdim",
            Path("/usr/local/fsl/bin/fslswapdim"),
            Path("/opt/fsl/bin/fslswapdim"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_triptych_preview(volume_path: Path, output_path: Path, panel_size: tuple[int, int]) -> None:
    volume = nib.load(str(volume_path)).get_fdata()
    slices = [slice_array for _, slice_array in build_middle_view_triptych(volume)]

    margin = 14
    width = panel_size[0] * len(slices) + CARD_SPACING * (len(slices) - 1) + (margin * 2)
    height = panel_size[1] + (margin * 2)
    canvas = Image.new("RGB", (width, height), COLORS["surface_soft"])

    for index, slice_array in enumerate(slices):
        tile = Image.fromarray(slice_array, mode="RGB")
        tile = ImageOps.contain(tile, panel_size, method=Image.Resampling.LANCZOS)

        panel = Image.new("RGB", panel_size, COLORS["surface"])
        left = (panel_size[0] - tile.width) // 2
        top = (panel_size[1] - tile.height) // 2
        panel.paste(tile, (left, top))

        x_offset = margin + index * (panel_size[0] + CARD_SPACING)
        canvas.paste(panel, (x_offset, margin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def detect_left_right_flip_warning(command_output: str) -> bool:
    normalized = " ".join(command_output.lower().split())
    warning_markers = (
        "left-right orientation is being flipped",
        "left/right orientation is being flipped",
        "flipping left/right orientation",
        "flipping left-right orientation",
    )
    return any(marker in normalized for marker in warning_markers)


def ensure_reference_preview_asset() -> Path:
    if BUNDLED_REFERENCE_PREVIEW.exists():
        return BUNDLED_REFERENCE_PREVIEW

    if not BUNDLED_REFERENCE_VOLUME.exists():
        raise FileNotFoundError(
            f"Bundled reference volume was not found at {BUNDLED_REFERENCE_VOLUME}."
        )

    build_triptych_preview(
        BUNDLED_REFERENCE_VOLUME,
        BUNDLED_REFERENCE_PREVIEW,
        REFERENCE_PANEL_SIZE,
    )
    return BUNDLED_REFERENCE_PREVIEW


class OrientationWorkspace:
    def __init__(self, input_path: Path, fslswapdim_path: Path) -> None:
        self.input_path = Path(input_path).expanduser().resolve()
        self.fslswapdim_path = Path(fslswapdim_path).expanduser().resolve()
        self.root_dir = Path(tempfile.mkdtemp(prefix="pigbet_orientation_"))
        self.preview_dir = self.root_dir / "previews"
        self.volume_dir = self.root_dir / "candidates"
        self.candidates: dict[str, CandidateResult] = {}
        self.cancel_event = threading.Event()

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.volume_dir.mkdir(parents=True, exist_ok=True)

    def _run_fslswapdim(self, spec: OrientationSpec, output_path: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("FSLDIR", str(self.fslswapdim_path.parent.parent))
        env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

        command = [
            str(self.fslswapdim_path),
            str(self.input_path),
            *spec.axes,
            str(output_path),
        ]
        try:
            return subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"fslswapdim failed for orientation '{spec.label}': {detail}") from exc

    def generate_candidates(self, emit: Callable[[dict], None] | None = None) -> None:
        total = len(ORIENTATION_SPECS)
        for index, spec in enumerate(ORIENTATION_SPECS, start=1):
            if self.cancel_event.is_set():
                break

            volume_path = self.volume_dir / f"{index:02d}_{spec.slug}.nii.gz"
            preview_path = self.preview_dir / f"{index:02d}_{spec.slug}.png"

            completed = self._run_fslswapdim(spec, volume_path)
            build_triptych_preview(volume_path, preview_path, CARD_PANEL_SIZE)
            command_output = f"{completed.stdout}\n{completed.stderr}".strip()

            candidate = CandidateResult(
                spec=spec,
                volume_path=volume_path,
                preview_path=preview_path,
                left_right_flip=detect_left_right_flip_warning(command_output),
                warning_text=command_output,
            )
            self.candidates[spec.label] = candidate

            if emit:
                emit(
                    {
                        "type": "candidate",
                        "candidate": candidate,
                        "index": index,
                        "total": total,
                    }
                )

        if emit:
            emit({"type": "done", "total": len(self.candidates)})

    def save_candidate(self, label: str, output_path: Path) -> Path:
        candidate = self.candidates[label]
        final_path = ensure_nii_gz_path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.volume_path, final_path)
        return final_path

    def cleanup(self) -> None:
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def cancel(self) -> None:
        self.cancel_event.set()


class CandidateCard(QFrame):
    clicked = Signal(str)

    def __init__(self, candidate: CandidateResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.candidate = candidate
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(CARD_WIDGET_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap(str(candidate.preview_path)))
        layout.addWidget(self.image_label)

        title_text = candidate.spec.label
        if candidate.spec.is_identity:
            title_text = f"{title_text}   original"

        self.title_label = QLabel(title_text)
        self.title_label.setWordWrap(True)
        self.title_label.setObjectName("cardTitle")
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("Sagittal  |  Coronal  |  Axial")
        self.subtitle_label.setObjectName("cardSubtitle")
        layout.addWidget(self.subtitle_label)

        self.warning_label = QLabel()
        self.warning_label.setObjectName("warningBadge")
        self.warning_label.setVisible(candidate.left_right_flip)
        if candidate.left_right_flip:
            self.warning_label.setText("L/R Flip")
            self.warning_label.setToolTip(
                candidate.warning_text or "fslswapdim reported that the left-right orientation is being flipped."
            )
        layout.addWidget(self.warning_label, 0, Qt.AlignLeft)

        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        border = COLORS["selected"] if selected else COLORS["border"]
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {COLORS['surface']};
                border: 2px solid {border};
                border-radius: 20px;
            }}
            QLabel {{
                background: transparent;
                border: none;
                color: {COLORS['text']};
            }}
            QLabel#cardTitle {{
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#cardSubtitle {{
                font-size: 11px;
                color: {COLORS['muted']};
            }}
            QLabel#warningBadge {{
                background: #fff1f2;
                color: #9f1239;
                border: 1px solid #fecdd3;
                border-radius: 10px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 700;
            }}
            """
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.candidate.spec.label)
        super().mousePressEvent(event)


class OrientationHelperWindow(QMainWindow):
    def __init__(
        self,
        initial_input: str | None = None,
        initial_output: str | None = None,
        legacy_reference: str | None = None,
        initial_fslswapdim: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 980)
        self.setMinimumSize(1180, 780)

        self.workspace: OrientationWorkspace | None = None
        self.worker: threading.Thread | None = None
        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.card_widgets: dict[str, CandidateCard] = {}
        self.card_order: list[str] = []
        self.selected_label: str | None = None
        self.is_generating = False

        self.reference_preview_path: Path | None = None
        self.setup_panel_visible = False

        self._build_ui(
            initial_input=initial_input,
            initial_output=initial_output,
            legacy_reference=legacy_reference,
            initial_fslswapdim=initial_fslswapdim,
        )

        try:
            self.reference_preview_path = ensure_reference_preview_asset()
            self._set_reference_preview(self.reference_preview_path)
        except Exception as exc:
            self.reference_preview_label.setText(f"Failed to load bundled reference preview.\n\n{exc}")

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_queue)
        self.poll_timer.start(120)

    def _build_ui(
        self,
        initial_input: str | None,
        initial_output: str | None,
        legacy_reference: str | None,
        initial_fslswapdim: str | None,
    ) -> None:
        del legacy_reference
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {COLORS['bg']};
            }}
            QWidget {{
                color: {COLORS['text']};
                font-family: "Avenir Next", "SF Pro Display", "Helvetica Neue", sans-serif;
            }}
            QLabel#title {{
                font-size: 32px;
                font-weight: 800;
            }}
            QLabel#subtitle, QLabel#muted {{
                color: {COLORS['muted']};
                font-size: 12px;
            }}
            QLabel#section {{
                font-size: 16px;
                font-weight: 700;
            }}
            QFrame#panel {{
                background: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 22px;
            }}
            QFrame#heroPanel {{
                background: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 28px;
            }}
            QLabel#fieldLabel {{
                color: {COLORS['muted']};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.04em;
            }}
            QLineEdit {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 14px;
                padding: 10px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent']};
            }}
            QPushButton {{
                border-radius: 14px;
                padding: 10px 16px;
                font-size: 12px;
                font-weight: 700;
                border: 1px solid {COLORS['border']};
                background: {COLORS['surface']};
            }}
            QPushButton:hover {{
                background: {COLORS['surface_soft']};
            }}
            QPushButton#accentButton {{
                background: {COLORS['accent']};
                color: white;
                border: none;
            }}
            QPushButton#accentButton:hover {{
                background: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background: {COLORS['surface_soft']};
                color: {COLORS['muted']};
            }}
            QProgressBar {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 11px;
                min-height: 22px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {COLORS['accent']};
                border-radius: 10px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QPushButton#ghostButton {{
                background: transparent;
                border: 1px solid {COLORS['border']};
                color: {COLORS['muted']};
            }}
            QPushButton#ghostButton:hover {{
                background: {COLORS['surface_soft']};
            }}
            QLabel#pathSummary {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 14px;
                padding: 10px 12px;
                color: {COLORS['muted']};
                font-size: 12px;
            }}
            QLabel#compareTitle {{
                font-size: 13px;
                font-weight: 700;
            }}
            QLabel#selectionValue {{
                font-size: 15px;
                font-weight: 700;
            }}
            """
        )

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(16)

        hero_panel = QFrame()
        hero_panel.setObjectName("heroPanel")
        hero_layout = QVBoxLayout(hero_panel)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(14)
        root_layout.addWidget(hero_panel)

        header_row = QHBoxLayout()
        header_row.setSpacing(14)
        hero_layout.addLayout(header_row)

        header_text = QVBoxLayout()
        header_text.setSpacing(4)
        header_row.addLayout(header_text, 1)

        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        header_text.addWidget(title)

        subtitle = QLabel(
            "Generate every fslswapdim orientation, compare their middle PigBET slice views against the reference, and save only the matching .nii.gz."
        )
        subtitle.setObjectName("subtitle")
        header_text.addWidget(subtitle)

        action_buttons = QHBoxLayout()
        action_buttons.setSpacing(10)
        header_row.addLayout(action_buttons)

        self.setup_toggle_button = QPushButton("Show Setup")
        self.setup_toggle_button.setObjectName("ghostButton")
        self.setup_toggle_button.clicked.connect(self._toggle_setup_panel)
        action_buttons.addWidget(self.setup_toggle_button)

        self.generate_button = QPushButton("Generate Options")
        self.generate_button.setObjectName("accentButton")
        self.generate_button.clicked.connect(self._start_generation)
        action_buttons.addWidget(self.generate_button)

        self.save_button = QPushButton("Save Selected .nii.gz")
        self.save_button.clicked.connect(self._save_selected_candidate)
        self.save_button.setEnabled(False)
        action_buttons.addWidget(self.save_button)

        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self._reset_results)
        action_buttons.addWidget(self.clear_button)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(12)
        hero_layout.addLayout(progress_row)

        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("muted")
        progress_row.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_bar, 1)

        self.controls_panel = QFrame()
        self.controls_panel.setObjectName("panel")
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(22, 20, 22, 20)
        controls_layout.setSpacing(14)
        hero_layout.addWidget(self.controls_panel)

        fslswapdim_path = resolve_fslswapdim_path(initial_fslswapdim)
        self.input_edit = self._build_path_row(controls_layout, "Input image", initial_input or "", self._browse_input_file)
        self.output_edit = self._build_path_row(controls_layout, "Output image", initial_output or "", self._browse_output_file)
        self.fslswapdim_edit = self._build_path_row(
            controls_layout,
            "fslswapdim binary",
            str(fslswapdim_path) if fslswapdim_path else "",
            self._browse_fslswapdim,
        )

        if self.input_edit.text().strip() and not self.output_edit.text().strip():
            self.output_edit.setText(str(derive_default_output_path(self.input_edit.text().strip())))

        self.input_edit.textChanged.connect(self._sync_output_path)
        self.output_edit.textChanged.connect(self._refresh_session_summary)
        self.fslswapdim_edit.textChanged.connect(self._refresh_session_summary)
        self.controls_panel.setVisible(False)

        split_view = QSplitter(Qt.Horizontal)
        split_view.setChildrenCollapsible(False)
        root_layout.addWidget(split_view, 1)

        browser_panel = QFrame()
        browser_panel.setObjectName("panel")
        browser_layout = QVBoxLayout(browser_panel)
        browser_layout.setContentsMargins(20, 20, 20, 20)
        browser_layout.setSpacing(12)
        split_view.addWidget(browser_panel)

        browser_header = QHBoxLayout()
        browser_header.setSpacing(12)
        browser_layout.addLayout(browser_header)

        browser_title_stack = QVBoxLayout()
        browser_title_stack.setSpacing(3)
        browser_header.addLayout(browser_title_stack, 1)

        candidates_title = QLabel("Candidates")
        candidates_title.setObjectName("section")
        browser_title_stack.addWidget(candidates_title)

        candidates_subtitle = QLabel(
            "Each card shows the same middle sagittal, coronal, and axial 2.5D slices PigBET uses during preprocessing."
        )
        candidates_subtitle.setObjectName("muted")
        browser_title_stack.addWidget(candidates_subtitle)

        self.candidate_count_label = QLabel("0 options")
        self.candidate_count_label.setObjectName("muted")
        browser_header.addWidget(self.candidate_count_label, 0, Qt.AlignRight | Qt.AlignTop)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.viewport().installEventFilter(self)
        browser_layout.addWidget(self.scroll_area, 1)

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setHorizontalSpacing(18)
        self.cards_layout.setVerticalSpacing(18)
        self.cards_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.cards_container)

        self.empty_state_label = QLabel("No orientation candidates yet.")
        self.empty_state_label.setObjectName("muted")
        self.cards_layout.addWidget(self.empty_state_label, 0, 0)

        self.sidebar_panel = QFrame()
        self.sidebar_panel.setObjectName("panel")
        self.sidebar_panel.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(self.sidebar_panel)
        sidebar_layout.setContentsMargins(18, 18, 18, 18)
        sidebar_layout.setSpacing(14)
        split_view.addWidget(self.sidebar_panel)

        sidebar_title = QLabel("Compare")
        sidebar_title.setObjectName("section")
        sidebar_layout.addWidget(sidebar_title)

        sidebar_subtitle = QLabel(
            "The reference and selected candidate use the same middle-slice preprocessing view for direct comparison."
        )
        sidebar_subtitle.setObjectName("muted")
        sidebar_layout.addWidget(sidebar_subtitle)

        reference_panel = QFrame()
        reference_panel.setObjectName("panel")
        reference_layout = QVBoxLayout(reference_panel)
        reference_layout.setContentsMargins(16, 16, 16, 16)
        reference_layout.setSpacing(10)
        sidebar_layout.addWidget(reference_panel)

        reference_title = QLabel("Reference")
        reference_title.setObjectName("compareTitle")
        reference_layout.addWidget(reference_title)

        self.reference_preview_label = QLabel("Loading reference preview...")
        self.reference_preview_label.setObjectName("muted")
        self.reference_preview_label.setAlignment(Qt.AlignCenter)
        reference_layout.addWidget(self.reference_preview_label)

        selected_panel = QFrame()
        selected_panel.setObjectName("panel")
        selected_layout = QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(16, 16, 16, 16)
        selected_layout.setSpacing(10)
        sidebar_layout.addWidget(selected_panel)

        selected_title = QLabel("Selected Candidate")
        selected_title.setObjectName("compareTitle")
        selected_layout.addWidget(selected_title)

        self.selected_orientation_label = QLabel("Nothing selected yet")
        self.selected_orientation_label.setObjectName("selectionValue")
        self.selected_orientation_label.setWordWrap(True)
        selected_layout.addWidget(self.selected_orientation_label)

        self.selected_preview_label = QLabel("Click a candidate card to compare it here.")
        self.selected_preview_label.setObjectName("muted")
        self.selected_preview_label.setAlignment(Qt.AlignCenter)
        selected_layout.addWidget(self.selected_preview_label)

        paths_panel = QFrame()
        paths_panel.setObjectName("panel")
        paths_layout = QVBoxLayout(paths_panel)
        paths_layout.setContentsMargins(16, 16, 16, 16)
        paths_layout.setSpacing(8)
        sidebar_layout.addWidget(paths_panel)

        paths_title = QLabel("Session")
        paths_title.setObjectName("compareTitle")
        paths_layout.addWidget(paths_title)

        self.input_summary_label = QLabel()
        self.input_summary_label.setObjectName("pathSummary")
        self.input_summary_label.setWordWrap(True)
        paths_layout.addWidget(self.input_summary_label)

        self.output_summary_label = QLabel()
        self.output_summary_label.setObjectName("pathSummary")
        self.output_summary_label.setWordWrap(True)
        paths_layout.addWidget(self.output_summary_label)

        self.fslswapdim_summary_label = QLabel()
        self.fslswapdim_summary_label.setObjectName("pathSummary")
        self.fslswapdim_summary_label.setWordWrap(True)
        paths_layout.addWidget(self.fslswapdim_summary_label)

        sidebar_layout.addStretch(1)

        split_view.setStretchFactor(0, 1)
        split_view.setStretchFactor(1, 0)

        self.status_label = QLabel("Choose an input volume to generate orientation candidates.")
        self.status_label.setObjectName("muted")
        root_layout.addWidget(self.status_label)

        self._refresh_session_summary()

    def _build_path_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        initial_text: str,
        browse_callback: Callable[[], None],
    ) -> QLineEdit:
        row_container = QWidget()
        row_layout = QVBoxLayout(row_container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        row_layout.addWidget(label)

        input_row = QHBoxLayout()
        input_row.setSpacing(12)
        row_layout.addLayout(input_row)

        line_edit = QLineEdit(initial_text)
        input_row.addWidget(line_edit, 1)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(browse_callback)
        input_row.addWidget(browse_button)

        parent_layout.addWidget(row_container)
        return line_edit

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self.scroll_area.viewport() and event.type() == QEvent.Resize:
            self._relayout_cards()
        return super().eventFilter(watched, event)

    def _toggle_setup_panel(self) -> None:
        self.setup_panel_visible = not self.setup_panel_visible
        self.controls_panel.setVisible(self.setup_panel_visible)
        self.setup_toggle_button.setText("Hide Setup" if self.setup_panel_visible else "Show Setup")

    def _sync_output_path(self) -> None:
        input_value = self.input_edit.text().strip()
        if not input_value:
            return

        current_output = self.output_edit.text().strip()
        if not current_output or current_output.endswith("_oriented.nii.gz"):
            self.output_edit.setText(str(derive_default_output_path(input_value)))
        self._refresh_session_summary()

    def _browse_input_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input NIfTI image",
            "",
            "NIfTI images (*.nii *.nii.gz);;All files (*)",
        )
        if selected:
            self.input_edit.setText(selected)

    def _browse_output_file(self) -> None:
        current = self.output_edit.text().strip()
        if current:
            initial_dir = str(Path(current).expanduser().parent)
            initial_name = Path(current).name
        else:
            default_path = derive_default_output_path(self.input_edit.text().strip() or "output.nii.gz")
            initial_dir = str(default_path.parent)
            initial_name = default_path.name

        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save reoriented image as",
            str(Path(initial_dir) / initial_name),
            "Compressed NIfTI (*.nii.gz);;All files (*)",
        )
        if selected:
            self.output_edit.setText(str(ensure_nii_gz_path(selected)))

    def _browse_fslswapdim(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose fslswapdim binary", "")
        if selected:
            self.fslswapdim_edit.setText(selected)
            self._refresh_session_summary()

    def _validate_paths(self) -> tuple[Path, Path, Path]:
        input_path = Path(self.input_edit.text().strip()).expanduser()
        if not input_path.exists():
            raise FileNotFoundError("The input image was not found.")
        if not (input_path.name.endswith(".nii") or input_path.name.endswith(".nii.gz")):
            raise ValueError("The input image must be a .nii or .nii.gz file.")

        output_path = ensure_nii_gz_path(self.output_edit.text().strip() or derive_default_output_path(input_path))
        fslswapdim_path = resolve_fslswapdim_path(self.fslswapdim_edit.text().strip() or None)
        if not fslswapdim_path:
            raise FileNotFoundError("fslswapdim was not found. Choose the binary explicitly.")

        return input_path, output_path, fslswapdim_path

    def _refresh_session_summary(self) -> None:
        input_text = self.input_edit.text().strip() or "Input: not set"
        output_text = self.output_edit.text().strip() or "Output: not set"
        fsl_text = self.fslswapdim_edit.text().strip() or "fslswapdim: not set"

        self.input_summary_label.setText(f"Input\n{input_text}")
        self.output_summary_label.setText(f"Output\n{output_text}")
        self.fslswapdim_summary_label.setText(f"fslswapdim\n{fsl_text}")

    def _build_message_box(
        self,
        icon: QMessageBox.Icon,
        text: str,
        buttons: QMessageBox.StandardButtons,
        default_button: QMessageBox.StandardButton | None = None,
    ) -> QMessageBox:
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(APP_TITLE)
        box.setText(text)
        box.setStandardButtons(buttons)
        if default_button is not None:
            box.setDefaultButton(default_button)
        box.setStyleSheet(
            f"""
            QMessageBox {{
                background: {COLORS['surface']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text']};
                font-size: 14px;
                min-width: 340px;
            }}
            QMessageBox QPushButton {{
                background: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 8px 18px;
                min-width: 88px;
                font-weight: 700;
            }}
            QMessageBox QPushButton:hover {{
                background: {COLORS['surface_soft']};
            }}
            """
        )
        return box

    def _show_error(self, text: str) -> None:
        self._build_message_box(
            QMessageBox.Critical,
            text,
            QMessageBox.Ok,
            QMessageBox.Ok,
        ).exec()

    def _show_info(self, text: str) -> None:
        self._build_message_box(
            QMessageBox.Information,
            text,
            QMessageBox.Ok,
            QMessageBox.Ok,
        ).exec()

    def _ask_yes_no(self, text: str, default: QMessageBox.StandardButton = QMessageBox.No) -> bool:
        result = self._build_message_box(
            QMessageBox.Question,
            text,
            QMessageBox.Yes | QMessageBox.No,
            default,
        ).exec()
        return result == QMessageBox.Yes

    def _set_reference_preview(self, preview_path: Path | None) -> None:
        if preview_path is None:
            self.reference_preview_label.setPixmap(QPixmap())
            self.reference_preview_label.setText("Reference preview is unavailable.")
            return

        pixmap = QPixmap(str(preview_path))
        if pixmap.isNull():
            self.reference_preview_label.setText("Failed to load the reference preview.")
            return

        self.reference_preview_label.setText("")
        self.reference_preview_label.setPixmap(
            pixmap.scaledToWidth(SIDEBAR_WIDTH - 70, Qt.SmoothTransformation)
        )

    def _set_selected_preview(self, preview_path: Path | None, orientation_label: str | None = None) -> None:
        if orientation_label:
            self.selected_orientation_label.setText(orientation_label)
        else:
            self.selected_orientation_label.setText("Nothing selected yet")

        if preview_path is None:
            self.selected_preview_label.setPixmap(QPixmap())
            self.selected_preview_label.setText("Click a candidate card to compare it here.")
            return

        pixmap = QPixmap(str(preview_path))
        if pixmap.isNull():
            self.selected_preview_label.setPixmap(QPixmap())
            self.selected_preview_label.setText("Failed to load the selected candidate preview.")
            return

        self.selected_preview_label.setText("")
        self.selected_preview_label.setPixmap(
            pixmap.scaledToWidth(SIDEBAR_WIDTH - 70, Qt.SmoothTransformation)
        )

    def _start_generation(self) -> None:
        if self.is_generating:
            return

        try:
            input_path, output_path, fslswapdim_path = self._validate_paths()
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.output_edit.setText(str(output_path))
        self._reset_results()

        self.workspace = OrientationWorkspace(input_path, fslswapdim_path)
        self.is_generating = True
        self.generate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating fslswapdim candidates from the input image...")
        self.progress_label.setText("Generating candidates")
        self._refresh_session_summary()

        def worker() -> None:
            try:
                assert self.workspace is not None
                self.workspace.generate_candidates(emit=self.event_queue.put)
            except Exception as exc:
                self.event_queue.put({"type": "error", "message": str(exc)})

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _poll_queue(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")
            if event_type == "candidate":
                candidate = event["candidate"]
                index = event["index"]
                total = event["total"]
                self._add_candidate_card(candidate)
                self.progress_bar.setValue(int(index / total * 100))
                self.progress_label.setText(f"{index}/{total} generated")
                self.status_label.setText(f"Generated {index}/{total}: {candidate.spec.label}")
            elif event_type == "done":
                count = event["total"]
                self.is_generating = False
                self.generate_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                self.save_button.setEnabled(self.selected_label is not None)
                self.progress_label.setText(f"{count} options ready")
                self.status_label.setText(f"Ready. Review {count} orientation options and save the matching .nii.gz.")
            elif event_type == "error":
                self.is_generating = False
                self.generate_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                self.save_button.setEnabled(False)
                self.progress_label.setText("Generation failed")
                self.status_label.setText("Generation failed.")
                self._show_error(event["message"])
                self._cleanup_workspace()

    def _add_candidate_card(self, candidate: CandidateResult) -> None:
        if self.empty_state_label.isVisible():
            self.empty_state_label.hide()

        card = CandidateCard(candidate)
        card.clicked.connect(self._select_card)
        self.card_widgets[candidate.spec.label] = card
        self.card_order.append(candidate.spec.label)
        self.candidate_count_label.setText(f"{len(self.card_order)} options")
        self._relayout_cards()

        if self.selected_label is None:
            self._select_card(candidate.spec.label)

    def _relayout_cards(self) -> None:
        if not self.card_order:
            self.empty_state_label.show()
            self.cards_layout.addWidget(self.empty_state_label, 0, 0)
            return

        self.cards_layout.removeWidget(self.empty_state_label)

        for label in self.card_order:
            self.cards_layout.removeWidget(self.card_widgets[label])

        viewport_width = max(1, self.scroll_area.viewport().width())
        columns = max(BROWSER_MIN_COLUMNS, viewport_width // (CARD_WIDGET_WIDTH + BROWSER_COLUMN_GAP))

        for index, label in enumerate(self.card_order):
            row = index // columns
            column = index % columns
            self.cards_layout.addWidget(self.card_widgets[label], row, column, Qt.AlignTop | Qt.AlignLeft)

    def _select_card(self, label: str) -> None:
        self.selected_label = label
        for current_label, card in self.card_widgets.items():
            card.set_selected(current_label == label)

        if not self.is_generating:
            self.save_button.setEnabled(True)
        candidate = self.workspace.candidates.get(label) if self.workspace else None
        preview_path = candidate.preview_path if candidate else None
        display_label = label
        if candidate and candidate.left_right_flip:
            display_label = f"{label}  |  L/R Flip"
        self._set_selected_preview(preview_path, display_label)
        self.status_label.setText(f"Selected orientation: {label}")

    def _save_selected_candidate(self) -> None:
        if not self.workspace or not self.selected_label:
            return

        final_path = ensure_nii_gz_path(
            self.output_edit.text().strip() or derive_default_output_path(self.input_edit.text().strip())
        )
        if final_path.exists():
            if not self._ask_yes_no(f"{final_path.name} already exists.\n\nOverwrite it?"):
                return

        try:
            saved_path = self.workspace.save_candidate(self.selected_label, final_path)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.status_label.setText(f"Saved {saved_path.name}. Temporary orientation files were cleaned up.")
        self._show_info(f"Saved reoriented image to:\n\n{saved_path}")
        self._cleanup_workspace()
        self.save_button.setEnabled(False)
        self.progress_label.setText("Saved selected output")

    def _reset_results(self) -> None:
        for card in self.card_widgets.values():
            self.cards_layout.removeWidget(card)
            card.deleteLater()

        self.card_widgets.clear()
        self.card_order.clear()
        self.selected_label = None
        self.progress_bar.setValue(0)
        self.progress_label.setText("Idle")
        self.save_button.setEnabled(False)
        self.empty_state_label.show()
        self.cards_layout.addWidget(self.empty_state_label, 0, 0)
        self.candidate_count_label.setText("0 options")
        self._set_selected_preview(None)

        if not self.is_generating:
            self._cleanup_workspace()

    def _cleanup_workspace(self) -> None:
        if self.workspace and not self.is_generating:
            self.workspace.cleanup()
            self.workspace = None

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self.is_generating:
            if not self._ask_yes_no(
                "Orientation generation is still running.\n\nClose the helper anyway?"
            ):
                event.ignore()
                return
            if self.workspace:
                self.workspace.cancel()
            event.accept()
            return

        self._cleanup_workspace()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive orientation helper for PigBET inputs.")
    parser.add_argument("--input", type=str, default=None, help="Optional input NIfTI image to preload.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path. Saved files are always .nii.gz.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fslswapdim",
        type=str,
        default=None,
        help="Optional explicit path to the fslswapdim binary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication.instance() or QApplication([])
    window = OrientationHelperWindow(
        initial_input=args.input,
        initial_output=args.output,
        legacy_reference=args.reference,
        initial_fslswapdim=args.fslswapdim,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
