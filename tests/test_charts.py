from __future__ import annotations

import unittest

from dlef.charts import BarChart, ChartBar, render_bar_chart_svg


class ChartRenderTests(unittest.TestCase):
    def test_bar_chart_contains_labels_values_and_best_marker(self) -> None:
        chart = BarChart(
            chart_id="demo",
            title="Validation MAE",
            subtitle="2 experiments",
            mode="min",
            bars=(
                ChartBar(label="baseline", value=0.5, note="1 run"),
                ChartBar(label="tuned", value=0.25, low=0.2, high=0.3, note="3 runs"),
            ),
        )
        svg = render_bar_chart_svg(chart)
        self.assertIn("<svg", svg)
        self.assertIn("Validation MAE", svg)
        self.assertIn("baseline", svg)
        self.assertIn("tuned", svg)
        self.assertIn("0.25", svg)
        self.assertIn('stroke="#111827"', svg)  # best bar outline
        self.assertIn("lower is better", svg)
        self.assertIn("<line", svg)  # grid / whisker
        self.assertIn("</svg>", svg)

    def test_labels_and_notes_are_escaped(self) -> None:
        chart = BarChart(
            chart_id="escape",
            title="<Title>",
            bars=(ChartBar(label="<script>", value=1.0, note="a & b"),),
        )
        svg = render_bar_chart_svg(chart)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("&lt;Title&gt;", svg)

    def test_empty_chart_has_empty_state(self) -> None:
        svg = render_bar_chart_svg(
            BarChart(chart_id="empty", title="Nothing", bars=())
        )
        self.assertIn("no data for this view", svg)

    def test_negative_values_do_not_crash(self) -> None:
        chart = BarChart(
            chart_id="negative",
            title="Delta",
            mode="max",
            bars=(
                ChartBar(label="a", value=-1.5),
                ChartBar(label="b", value=2.0),
            ),
        )
        svg = render_bar_chart_svg(chart)
        self.assertIn("-1.5", svg)
        self.assertIn("higher is better", svg)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_bar_chart_svg(
                BarChart(chart_id="bad", title="bad", mode="sideways")
            )


if __name__ == "__main__":
    unittest.main()
