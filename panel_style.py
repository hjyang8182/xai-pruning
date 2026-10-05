"""Text fitting for the narrow paper panels (plot_effective_concept_size.py,
plot_concept_reduction.py).

Those panels are deliberately crammed: a small width with large type, so they survive
the shrink to one column of a two-column figure. At that size the *text* sets the
figure's real width -- matplotlib does not clip text, and `bbox_inches="tight"` grows
the saved figure to include anything that overhangs the axes. So every long string has
to be measured against the space it actually has and wrapped (or shortened) to fit,
rather than trusted to a character count.
"""
import os
import textwrap


def text_width_in(fig, text, fontsize):
    """Width of `text` in inches if drawn on `fig` at `fontsize` (longest line)."""
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for line in str(text).split("\n"):
        t = fig.text(0, 0, line, fontsize=fontsize)
        widest = max(widest, t.get_window_extent(renderer).width / fig.dpi)
        t.remove()
    return widest


def wrap_to_width(fig, text, fontsize, max_in):
    """`text` wrapped on word boundaries so no line is wider than `max_in` inches.

    Falls back to the narrowest wrap available when a single word is too long (a word is
    never broken -- "non-zero" splitting across lines reads worse than an overhang)."""
    words = str(text).split()
    if not words:
        return str(text)
    # Start from a character estimate, then tighten until it measures small enough.
    cols = max(4, len(text))
    while cols > 4:
        wrapped = "\n".join(textwrap.wrap(text, width=cols, break_on_hyphens=False))
        if text_width_in(fig, wrapped, fontsize) <= max_in:
            return wrapped
        cols = int(cols * 0.85) if int(cols * 0.85) < cols else cols - 1
    return "\n".join(textwrap.wrap(text, width=max(4, len(max(words, key=len))),
                                   break_on_hyphens=False))


def fit_one_line(fig, candidates, fontsize, max_in):
    """First string in `candidates` that fits `max_in` inches; the last one otherwise.

    Used where wrapping would look wrong (a value annotation on a bar): the caller
    passes the full label first and progressively shorter forms after it."""
    for c in candidates:
        if text_width_in(fig, c, fontsize) <= max_in:
            return c
    return candidates[-1]


# Paper figures are set in Spectral (the serif the write-up uses). It is not a system
# font, so it is installed under ~/.local/share/fonts and registered here; if it is
# missing (another machine, a fresh checkout) fall back through the serifs that ship
# with matplotlib rather than silently reverting to the sans default.
FONT_STACK = ["Spectral", "Source Serif Pro", "Noto Serif", "STIXGeneral",
              "DejaVu Serif", "serif"]


def use_paper_font(plt):
    """Point matplotlib at Spectral (or the closest serif available)."""
    import glob
    import matplotlib.font_manager as fm

    known = {f.name for f in fm.fontManager.ttflist}
    if "Spectral" not in known:
        for path in glob.glob(os.path.expanduser("~/.local/share/fonts/Spectral-*.ttf")):
            fm.fontManager.addfont(path)
        known = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [f for f in FONT_STACK if f in known or f == "serif"]
    return plt.rcParams["font.serif"][0]
