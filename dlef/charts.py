"""Static, dependency-free charts for experiment reports.

Charts are a derived projection of RunStore / candidates / claims, never a
record. The core only knows labels, values and SVG; which metric to draw and
which runs to compare belongs to the project's view spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

DEFAULT_BAR_COLOR = "#4C78A8"
BEST_BAR_COLOR = "#1D4E74"
INK = "#1F2933"
MUTED = "#66727F"
GRID = "#E3E8EE"
BACKGROUND = "#FFFFFF"


@dataclass(frozen=True)
class ChartBar:
    """One horizontal bar, optionally with a low/high whisker."""

    label: str
    value: float
    low: float | None = None
    high: float | None = None
    note: str = ""
    color: str = ""


@dataclass(frozen=True)
class BarChart:
    """A single horizontal bar chart."""

    chart_id: str
    title: str
    bars: tuple[ChartBar, ...] = ()
    subtitle: str = ""
    unit: str = ""
    mode: str = "min"  # min: lower is better; max: higher is better
    footer: str = ""


def render_bar_chart_svg(
    chart: BarChart,
    *,
    width: int = 1000,
    row_height: int = 42,
) -> str:
    """Render ``chart`` as a standalone SVG string."""

    if chart.mode not in {"min", "max"}:
        raise ValueError("chart mode must be 'min' or 'max'")

    width = max(480, int(width))
    row_height = max(28, int(row_height))
    bars = [bar for bar in chart.bars if _finite(bar.value)]
    best_index = _best_index(bars, chart.mode)

    label_width = min(260, max(170, width // 4))
    plot_x = 18 + label_width
    value_width = 96
    plot_width = max(80, width - plot_x - value_width - 18)
    top = 92
    bottom = 46
    height = top + max(1, len(bars)) * row_height + bottom

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(chart.title, quote=True)}">',
        f"<title>{escape(chart.title)}</title>",
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        f'<text x="18" y="32" font-family="{_font()}" font-size="20" '
        f'font-weight="600" fill="{INK}">{escape(chart.title)}</text>',
    ]
    if chart.subtitle:
        lines.append(
            f'<text x="18" y="54" font-family="{_font()}" font-size="12" '
            f'fill="{MUTED}">{escape(chart.subtitle)}</text>'
        )
    hint = "lower is better" if chart.mode == "min" else "higher is better"
    if chart.unit:
        hint += f" · unit: {chart.unit}"
    lines.append(
        f'<text x="18" y="72" font-family="{_font()}" font-size="11" '
        f'fill="{MUTED}">{escape(hint)}</text>'
    )

    if not bars:
        lines.append(
            f'<text x="{width // 2}" y="{top + row_height}" text-anchor="middle" '
            f'font-family="{_font()}" font-size="13" fill="{MUTED}">'
            "no data for this view</text>"
        )
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    low = min(min(_bar_low(bar) for bar in bars), 0.0)
    high = max(max(_bar_high(bar) for bar in bars), 0.0)
    if high == low:
        high = low + 1.0

    def x(value: float) -> float:
        return plot_x + (value - low) / (high - low) * plot_width

    zero_x = x(0.0)
    chart_bottom = top + len(bars) * row_height

    # Grid and axis labels.
    for step in range(5):
        value = low + (high - low) * step / 4
        gx = x(value)
        lines.append(
            f'<line x1="{gx:.2f}" y1="{top - 8}" x2="{gx:.2f}" '
            f'y2="{chart_bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{gx:.2f}" y="{chart_bottom + 20}" text-anchor="middle" '
            f'font-family="{_font()}" font-size="10" fill="{MUTED}">'
            f"{_format(value)}</text>"
        )
    lines.append(
        f'<line x1="{zero_x:.2f}" y1="{top - 8}" x2="{zero_x:.2f}" '
        f'y2="{chart_bottom}" stroke="{MUTED}" stroke-width="1.2"/>'
    )

    for index, bar in enumerate(bars):
        y = top + index * row_height
        bar_y = y + 15
        bar_h = 16
        is_best = index == best_index
        color = bar.color or (BEST_BAR_COLOR if is_best else DEFAULT_BAR_COLOR)
        stroke = f' stroke="#111827" stroke-width="2"' if is_best else ""
        start = min(zero_x, x(bar.value))
        bar_width = max(1.0, abs(x(bar.value) - zero_x))
        lines.append(
            f'<text x="18" y="{y + 18}" font-family="{_font()}" font-size="12" '
            f'fill="{INK}">{escape(_clip(bar.label, 34))}'
            f"<title>{escape(bar.label)}</title></text>"
        )
        if bar.note:
            lines.append(
                f'<text x="18" y="{y + 33}" font-family="{_font()}" font-size="10" '
                f'fill="{MUTED}">{escape(_clip(bar.note, 38))}</text>'
            )
        lines.append(
            f'<rect x="{start:.2f}" y="{bar_y}" width="{bar_width:.2f}" '
            f'height="{bar_h}" rx="2" fill="{escape(color)}"{stroke}/>'
        )

        low_value = _bar_low(bar)
        high_value = _bar_high(bar)
        if high_value > low_value:
            wx_low = x(low_value)
            wx_high = x(high_value)
            wy = bar_y + bar_h / 2
            lines.append(
                f'<line x1="{wx_low:.2f}" y1="{wy:.2f}" x2="{wx_high:.2f}" '
                f'y2="{wy:.2f}" stroke="{INK}" stroke-width="1.2"/>'
            )
            for wx in (wx_low, wx_high):
                lines.append(
                    f'<line x1="{wx:.2f}" y1="{wy - 4:.2f}" x2="{wx:.2f}" '
                    f'y2="{wy + 4:.2f}" stroke="{INK}" stroke-width="1.2"/>'
                )

        value_text = _format(bar.value)
        if bar.value < 0:
            tx, anchor = x(bar.value) - 6, "end"
        else:
            tx, anchor = x(bar.value) + 6, "start"
        lines.append(
            f'<text x="{tx:.2f}" y="{bar_y + 13}" text-anchor="{anchor}" '
            f'font-family="{_font()}" font-size="11" font-weight="600" '
            f'fill="{INK}">{value_text}</text>'
        )

    if chart.footer:
        lines.append(
            f'<text x="18" y="{height - 16}" font-family="{_font()}" '
            f'font-size="10" fill="{MUTED}">{escape(chart.footer)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _best_index(bars: list[ChartBar], mode: str) -> int | None:
    if not bars:
        return None
    values = [bar.value for bar in bars]
    target = min(values) if mode == "min" else max(values)
    return values.index(target)


def _bar_low(bar: ChartBar) -> float:
    return bar.low if bar.low is not None else bar.value


def _bar_high(bar: ChartBar) -> float:
    return bar.high if bar.high is not None else bar.value


def _finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}


def _format(value: float) -> str:
    return f"{value:.4g}"


def _clip(value: str, width: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _font() -> str:
    return "ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
