"""Stat card renderer: the shareable, download-friendly match summary.

AGENTS.md: "a downloadable, shareable stat card that gives you a score
(timeline plot + headline stats + scoring system for dopamine hit)" and
"stat card generated with PIL (readable at screen-share zoom)".

The card is composed with PIL: a dark 1600x1000 canvas with the match title,
a big score badge, the headline stats grid, an embedded timeline plot
(matplotlib, rendered to PNG first), and a footer stamp. Every element is
large enough to stay readable on a shared screen.
"""

from pathlib import Path

# ── Layout constants (1600 x 1000 card, generous for screen share) ───────────

CARD_WIDTH = 1600
CARD_HEIGHT = 1000

_BG = (16, 20, 27)
_PANEL = (27, 32, 42)
_TEXT = (238, 241, 246)
_MUTED = (150, 157, 170)
_GREEN = (46, 204, 113)
_RED = (231, 76, 60)
_AMBER = (241, 196, 15)
_BLUE = (52, 152, 219)
_PURPLE = (155, 89, 182)
_GRAY = (120, 127, 140)

# Event type -> (matplotlib marker, card color, label).
_EVENT_STYLE = {
    "hit_landed": ("o", _GREEN, "Hit landed"),
    "hit_taken": ("^", _RED, "Hit taken"),
    "punish": ("s", _AMBER, "Punish"),
    "whiff": ("x", _GRAY, "Whiff"),
    "round_win": ("*", _GREEN, "Round win"),
    "round_loss": ("*", _RED, "Round loss"),
    "domain_alert": ("D", _PURPLE, "Domain"),
}


def _font(size: int):
    """A bold-ish DejaVu font at *size*, falling back to the PIL default."""
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _score_band(score: float) -> tuple[str, tuple[int, int, int]]:
    """Headline verdict + badge color for a 0-100 score."""
    if score >= 70:
        return "DOMINANT", _GREEN
    if score >= 45:
        return "CLOSE CALL", _AMBER
    return "BEHIND", _RED


def _draw_stat_cell(draw, x, y, w, h, title: str, value: str, color=_TEXT):
    """One headline stat cell (label + big value) on the card."""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=_PANEL)
    draw.text((x + 24, y + 16), title.upper(), font=_font(20), fill=_MUTED)
    draw.text((x + 24, y + 46), value, font=_font(34), fill=color)


def _mean_enemy_health(state) -> float:
    enemies = state.get("enemies", []) if isinstance(state, dict) else []
    healths = [
        float(e) if isinstance(e, (int, float)) else float(e.get("health", 0.0))
        for e in enemies
    ]
    return float(sum(healths) / len(healths)) if healths else 0.0


def _mpl_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a 0-255 PIL colour to a normalized matplotlib colour."""
    return tuple(c / 255.0 for c in color)


def _plot_timeline(
    states: list[dict],
    events: list,
    save_path: str,
    round_boundaries: list[float],
) -> None:
    """Render the health curves + event markers into a matplotlib PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    xs = [float(s.get("frame_index", i)) for i, s in enumerate(states)]
    player = [float(s.get("player_health", 0.0)) for s in states]
    enemy = [_mean_enemy_health(s) for s in states]

    fig, ax = plt.subplots(figsize=(15.2, 4.4), dpi=100)
    fig.patch.set_facecolor("#14181f")
    ax.set_facecolor("#14181f")

    ax.plot(xs, player, color="#2ecc71", lw=2.2, label="Player health", zorder=2)
    ax.plot(xs, enemy, color="#e74c3c", lw=2.2, label="Enemy health", zorder=2)

    for r in round_boundaries:
        ax.axvline(r, color="#7f8c9d", ls="--", lw=1.1, alpha=0.8, zorder=1)

    # One marker row: hits on the victim's health line keeps the plot readable.
    seen: set[str] = set()
    for event in events:
        if event.frame_index > (xs[-1] if xs else 0):
            continue
        style = _EVENT_STYLE.get(event.type)
        if style is None:
            continue
        marker, _, label = style
        if event.type in ("hit_landed", "punish", "round_win"):
            y = enemy[event.frame_index] if event.frame_index < len(enemy) else 0.5
        else:
            y = player[event.frame_index] if event.frame_index < len(player) else 0.5
        ax.scatter(
            event.frame_index,
            y,
            marker=marker,
            s=110,
            color=_mpl_color(style[1]),
            zorder=3,
        )
        if event.type not in seen:
            seen.add(event.type)
            ax.scatter([], [], marker=marker, s=110, color=_mpl_color(style[1]), label=label)

    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Frame", color="#96a0ac")
    ax.set_ylabel("Health", color="#96a0ac")
    ax.tick_params(colors="#96a0ac")
    for spine in ax.spines.values():
        spine.set_color("#2a313d")
    ax.legend(facecolor="#1c222c", edgecolor="#2a313d", labelcolor="#eef1f6", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def render_stat_card(
    *,
    states: list[dict],
    events: list,
    output_path: str,
    vod_name: str = "match",
    score: float | None = None,
    headline: dict | None = None,
    run_id: str | None = None,
) -> str:
    """Render the full stat card PNG and return its path.

    Parameters
    ----------
    states
        The per-frame state dicts (for the timeline plot).
    events
        The gold event objects (detect_events output).
    output_path
        Where the PNG is written (parent dir is created).
    vod_name
        Match/VOD display name.
    score
        Computed 0-100 score; recomputed from the events when ``None``.
    headline
        Precomputed headline stats; recomputed when ``None``.
    run_id
        MLflow run id stamped at the bottom of the card.
    """
    from PIL import Image, ImageDraw

    if headline is None or score is None:
        # Lazy import: report.py imports render_stat_card at module level, so a
        # top-level import here would be circular.
        from src.pipeline.report import build_headline, compute_score

    if headline is None:
        headline = build_headline(events)
    if score is None:
        score = float(compute_score(headline))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Timeline plot -> temp PNG next to the card.
    round_starts: list[float] = []
    prev_round = None
    for s in states:
        r = s.get("round_index", 0) if isinstance(s, dict) else getattr(s, "round_index", 0)
        if prev_round is not None and r != prev_round:
            round_starts.append(float(s.get("frame_index", 0)))
        prev_round = r
    plot_path = out.with_suffix(".plot.png")
    _plot_timeline(states, events, str(plot_path), round_starts)

    # 2. Compose the card with PIL.
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), _BG)
    draw = ImageDraw.Draw(card)
    band, band_color = _score_band(score)

    draw.text((60, 44), "MATCH REPORT", font=_font(30), fill=_MUTED)
    draw.text((60, 88), vod_name, font=_font(52), fill=_TEXT)

    # Score badge.
    badge_x, badge_y, badge_w, badge_h = 60, 210, 320, 250
    draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=20, fill=_PANEL)
    draw.ellipse([badge_x + 90, badge_y + 22, badge_x + 230, badge_y + 162], outline=band_color, width=6)
    score_text = f"{round(score):.0f}"
    draw.text((badge_x + 100, badge_y + 58), score_text, font=_font(72), fill=band_color)
    draw.text((badge_x + 76, badge_y + 176), band, font=_font(30), fill=band_color)
    draw.text((badge_x + 58, badge_y + 212), "out of 100", font=_font(18), fill=_MUTED)

    # Headline stats grid (4 cols x 2 rows).
    stat_w, stat_h, gap = 280, 118, 18
    stat_x0 = 430
    stat_y0 = 210
    cells = [
        ("Hits Landed", str(headline["hits_landed"]), _GREEN),
        ("Hits Taken", str(headline["hits_taken"]), _RED),
        ("Punishes", str(headline["punishes"]), _AMBER),
        ("Whiffs", str(headline["whiffs"]), _GRAY),
        ("Rounds Won", str(headline["round_wins"]), _GREEN),
        ("Rounds Lost", str(headline["round_losses"]), _RED),
        ("Domains", str(headline["domain_alerts"]), _PURPLE),
        ("Damage", f"deal {headline['damage_dealt']:.0%}  take {headline['damage_taken']:.0%}", _TEXT),
    ]
    for idx, (title, value, color) in enumerate(cells):
        col, row = idx % 4, idx // 4
        x = stat_x0 + col * (stat_w + gap)
        y = stat_y0 + row * (stat_h + gap)
        _draw_stat_cell(draw, x, y, stat_w, stat_h, title, value, color)

    # Timeline plot.
    plot = Image.open(plot_path)
    scale = min((CARD_WIDTH - 120) / plot.width, 1.0)
    plot = plot.resize((int(plot.width * scale), int(plot.height * scale)), Image.LANCZOS)
    card.paste(plot, (60, 560))

    # Footer stamp.
    stamp = "Game State Vision Predictor  |  gold layer"
    if run_id:
        stamp += f"  |  run {run_id[:12]}"
    draw.text((60, CARD_HEIGHT - 52), stamp, font=_font(20), fill=_MUTED)

    card.save(out, format="PNG")
    plot_path.unlink(missing_ok=True)
    return str(out)


__all__ = ["CARD_HEIGHT", "CARD_WIDTH", "render_stat_card"]