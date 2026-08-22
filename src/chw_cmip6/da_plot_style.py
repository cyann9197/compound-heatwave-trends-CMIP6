"""Shared typography for the dynamic-adjustment figures.

This module is deliberately independent of the data-loading context so that
Fig. 6 and Extended Data Fig. 8 can be redrawn from prepared CSV files alone.
"""

PANEL_LABEL_X_WITH_TITLE = -0.06
PANEL_LABEL_Y_WITH_TITLE = 1.045


def _scaled_pt(base_pt: float, scale: float) -> float:
    return round(float(base_pt) * float(scale), 1)


FIG567_FONT_SCALE = 1.3
FIG567_PANEL_LABEL_PT = _scaled_pt(23, FIG567_FONT_SCALE)
FIG567_PANEL_TITLE_PT = _scaled_pt(19, FIG567_FONT_SCALE)
FIG567_AXIS_LABEL_PT = _scaled_pt(16, FIG567_FONT_SCALE)
FIG567_AXIS_TICK_PT = _scaled_pt(14.5, FIG567_FONT_SCALE)
FIG567_LEGEND_PT = _scaled_pt(12, FIG567_FONT_SCALE)
FIG567_CBAR_LABEL_PT = _scaled_pt(14, FIG567_FONT_SCALE)
FIG567_ANNOTATION_PT = _scaled_pt(14, FIG567_FONT_SCALE)

FINAL_DA_RCPARAMS = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": FIG567_AXIS_LABEL_PT,
    "axes.titlesize": FIG567_PANEL_TITLE_PT,
    "axes.labelsize": FIG567_AXIS_LABEL_PT,
    "xtick.labelsize": FIG567_AXIS_TICK_PT,
    "ytick.labelsize": FIG567_AXIS_TICK_PT,
    "legend.fontsize": FIG567_LEGEND_PT,
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
