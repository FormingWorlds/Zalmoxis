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
        for t in (ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()):
            if t.get_text() and t.get_visible():
                texts.append(t.get_window_extent(rnd))
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
