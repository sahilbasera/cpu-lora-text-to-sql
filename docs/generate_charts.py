"""Rebuild the four README SVG charts from the recorded v1 reports (stdlib only)."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "images"
RUNS = ROOT / "results"
BASE_RUN = "base_test_20260923T051755848526Z"
LORA_RUN = "lora_test_20260923T052011979719Z"
TRAIN_RUN = "sft_v1_20260923T045558219043Z"
PILOT_BEFORE = "base_pilot_20260904T235125761604Z"
PILOT_AFTER = "base_pilot_20260911T040037001999Z"
BASE, LORA = "#64748b", "#0f766e"
INK, MUTED, GRID = "#172b3a", "#526374", "#dce4ea"
WIDTH = 800
ET.register_namespace("", "http://www.w3.org/2000/svg")


def report(run, filename="summary.json"):
    return json.loads((RUNS / run / filename).read_text(encoding="utf-8"))


class Chart:
    def __init__(self, title, subtitle, height, description):
        self.svg = ET.Element("{http://www.w3.org/2000/svg}svg", {
            "viewBox": f"0 0 {WIDTH} {height}", "width": str(WIDTH),
            "height": str(height), "role": "img", "aria-labelledby": "title desc",
        })
        self.add("title", {"id": "title"}, title)
        self.add("desc", {"id": "desc"}, description)
        self.add("rect", {"width": WIDTH, "height": height, "fill": "#ffffff"})
        self.text(32, 40, title, size=24, weight=600)
        self.text(32, 67, subtitle, color=MUTED)

    def add(self, tag, attrs, text=None):
        element = ET.SubElement(self.svg, tag, {k: str(v) for k, v in attrs.items()})
        element.text = text
        return element

    def text(self, x, y, text, size=16, color=INK, anchor="start", weight=400):
        self.add("text", {"x": x, "y": y, "fill": color, "font-size": size,
                         "font-family": "Arial, sans-serif", "text-anchor": anchor,
                         "font-weight": weight}, str(text))

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.add("line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "stroke": color, "stroke-width": width})

    def bar(self, x, y, width, height, color):
        self.add("rect", {"x": x, "y": y, "width": width, "height": height, "fill": color})

    def save(self, filename):
        OUTPUT.mkdir(exist_ok=True)
        ET.indent(self.svg)
        path = OUTPUT / filename
        ET.ElementTree(self.svg).write(path, encoding="utf-8", xml_declaration=True)
        ET.parse(path)  # Ensure every generated asset remains well-formed XML.
        print(path.relative_to(ROOT))


def accuracy_chart(filename, title, subtitle, rows, footer):
    description = "; ".join(f"{label}: {r['correct']}/{r['total']}, {r['accuracy']:.1%}"
                            for label, r, _ in rows)
    chart = Chart(title, subtitle, 340, description + ". " + footer)
    for i, (label, result, color) in enumerate(rows):
        y = 110 + i * 77
        chart.text(32, y, label)
        chart.text(768, y, f"{result['correct']}/{result['total']}  |  {result['accuracy']:.1%}", anchor="end")
        chart.bar(32, y + 12, 736, 25, "#edf1f4")
        chart.bar(32, y + 12, 736 * result["accuracy"], 25, color)
    for value in (0, 25, 50, 75, 100):
        chart.text(32 + 7.36 * value, 248, f"{value}%", size=14,
                   anchor="start" if value == 0 else "end" if value == 100 else "middle", color=MUTED)
    chart.text(400, 275, "SQL execution accuracy · higher is better", anchor="middle", color=MUTED)
    chart.text(32, 316, footer)
    chart.save(filename)


def epoch_chart(training):
    epochs = training["epochs"]
    losses = [entry["validation_loss"] for entry in epochs]
    chart = Chart("Why epoch 1 was selected", "Validation loss · lower is better · post-training zoom", 405,
                  "; ".join(f"Epoch {e['epoch']}: {e['validation_loss']:.6f}" for e in epochs)
                  + ". The zoomed vertical axis does not start at zero.")
    low, high = min(losses), max(losses)
    pad = (high - low) * 0.4 or 0.001
    low, high = low - pad, high + pad
    left, right, top, bottom = 105, 749, 108, 289
    y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    xs = [left + 35 + i * (right - left - 70) / max(1, len(epochs) - 1) for i in range(len(epochs))]
    for i in range(4):
        value = low + i * (high - low) / 3
        chart.line(left, y(value), right, y(value))
        chart.text(left - 12, y(value) + 5, f"{value:.4f}", size=14, anchor="end", color=MUTED)
    chart.text(32, 93, "Loss", size=14, color=MUTED)
    chart.add("polyline", {"points": " ".join(f"{x},{y(v)}" for x, v in zip(xs, losses)),
                           "fill": "none", "stroke": LORA, "stroke-width": 3})
    for x, epoch in zip(xs, epochs):
        value = epoch["validation_loss"]
        selected = epoch["epoch"] == training["best_epoch"]
        chart.add("circle", {"cx": x, "cy": y(value), "r": 7 if selected else 5, "fill": LORA})
        chart.text(x, y(value) - 17, f"{value:.6f}", anchor="middle")
        chart.text(x, bottom + 27, f"Epoch {epoch['epoch']}", anchor="middle")
        if selected:
            chart.text(x, y(value) + 27, "Selected", color=LORA, anchor="middle", weight=600)
    chart.text(32, 360, f"Before training: {training['baseline_validation_loss']:.6f}")
    chart.text(32, 386, "Zoomed loss axis, not accuracy. Test accuracy was measured only for the selected adapter.", size=14, color=MUTED)
    chart.save("validation-loss.svg")


def family_chart(base, lora):
    families = [("country_eq", "Country equality"), ("amount_le", "Amount ≤ threshold"),
                ("amount_ge", "Amount ≥ threshold"), ("amount_lt", "Amount < threshold"),
                ("amount_gt", "Amount > threshold"), ("amount_range", "Inclusive range"),
                ("country_amount_gt", "Country + amount")]
    assert set(base["by_family"]) == set(lora["by_family"]) == {key for key, _ in families}
    assert all(r["total"] == 12 for s in (base, lora) for r in s["by_family"].values())
    chart = Chart("Where the gains happened", "Final test · 12 questions per family", 695,
                  "; ".join(f"{name}: base {base['by_family'][key]['correct']}, LoRA {lora['by_family'][key]['correct']} of 12"
                            for key, name in families))
    left, right, top, bottom = 282, 718, 110, 586
    for tick in (0, 3, 6, 9, 12):
        x = left + (right-left)*tick/12
        chart.line(x, top, x, bottom)
        chart.text(x, bottom+24, tick, size=14, anchor="middle", color=MUTED)
    for i, (key, name) in enumerate(families):
        y = top + i * 68
        chart.text(32, y+26, name)
        for j, (label, summary, color) in enumerate((("Base",base,BASE),("LoRA",lora,LORA))):
            correct = summary["by_family"][key]["correct"]
            chart.text(left-10, y+j*25+17, label, size=14, anchor="end", color=MUTED)
            chart.bar(left, y+j*25+3, (right-left)*correct/12, 18, color)
            chart.text(left+(right-left)*correct/12+8, y+j*25+17, f"{correct}/12", size=14)
    chart.text(500, 638, "Correct answers · higher is better", anchor="middle", color=MUTED)
    chart.text(32, 677, "Equal family totals can hide individual fixes and regressions.", size=14, color=MUTED)
    chart.save("accuracy-by-family.svg")


def main():
    base, lora = report(BASE_RUN), report(LORA_RUN)
    comparison = report(LORA_RUN, "comparison.json")
    assert base["total"] == lora["total"] == comparison["total"]
    assert (base["correct"], lora["correct"]) == (comparison["base_correct"], comparison["lora_correct"])
    outcomes = comparison["paired_outcomes"]
    gain = 100 * (lora["accuracy"] - base["accuracy"])
    accuracy_chart("base-vs-lora.svg", "LoRA improved final-test accuracy",
                   f"Same {base['total']} questions · validation-selected adapter",
                   [("Base model", base, BASE), ("LoRA SFT", lora, LORA)],
                   f"+{gain:.1f} percentage points · {outcomes['fixed']} fixes / {outcomes['regression']} regressions")
    epoch_chart(report(TRAIN_RUN, "training.json"))
    accuracy_chart("pilot-wording.svg", "Clearer questions, unchanged model",
                   "42-question development pilot · no fine-tuning",
                   [("Original wording", report(PILOT_BEFORE), BASE),
                    ("Clarified range wording", report(PILOT_AFTER), BASE)],
                   "Pilot and final test contain different questions.")
    family_chart(base, lora)


if __name__ == "__main__":
    main()
