"""Layout check for validation plots: no legend or axis text on a line vertex."""

from __future__ import annotations

import numpy as np


def assert_text_clear(fig):
    """Assert that no legend, axis label or tick label of ``fig`` holds a line vertex in display
    coordinates, and that no legend crosses an axis text; return ``fig``."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    legends = [lg.get_window_extent(rnd) for lg in fig.legends]
    texts, pts = [], []
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legends.append(ax.get_legend().get_window_extent(rnd))
        shown = [ax.xaxis.label, ax.yaxis.label]
        for axis in (ax.xaxis, ax.yaxis):  # matplotlib also lays out labels outside the view
            lo, hi = sorted(axis.get_view_interval())
            shown += [t.label1 for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi]
        texts += [t.get_window_extent(rnd) for t in shown if t.get_text() and t.get_visible()]
        for line in ax.get_lines():
            xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
            pts.append(xy[np.all(np.isfinite(xy), axis=1)])
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    for k, b in enumerate(legends + texts):
        inside = (
            (pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)
        )
        assert not inside.any(), f'text box {k} holds a line vertex'
    for lg in legends:
        assert not any(lg.overlaps(t) for t in texts), 'a legend crosses an axis text'
    return fig


def legend_below(fig, ncol=2, fontsize='small'):
    """Put one figure legend below the axes, with the layout leaving room for its height."""
    legend = fig.legend(loc='lower center', ncol=ncol, fontsize=fontsize)
    fig.canvas.draw()
    height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, height + 0.03, 1, 1))
    return fig
