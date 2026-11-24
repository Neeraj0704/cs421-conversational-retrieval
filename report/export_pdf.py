"""Presentation-only PDF export of the generated Markdown report.

This keeps the frozen inference/evaluation source unchanged.
"""

from pathlib import Path
from html import escape
import re
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    CondPageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "report/PROJECT_REPORT.md").read_text().splitlines()
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13.8
    styles["BodyText"].spaceAfter = 9
    styles["Heading2"].fontSize = 13.5
    styles["Heading2"].leading = 17
    styles["Heading2"].keepWithNext = True
    styles["Title"].fontSize = 19
    styles["Title"].leading = 23
    story = []
    i = 0

    def short(value):
        match = re.match(r"^(FINAL_)?([BER]\d\d)_", value)
        return ("Final " if match.group(1) else "") + match.group(2) if match else value

    while i < len(source):
        line = source[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if line.startswith("## "):
                story.append(CondPageBreak(155))
            story.append(
                Paragraph(
                    escape(title),
                    styles["Heading2" if line.startswith("## ") else "Title"],
                )
            )
            if title == "Development results":
                story.append(
                    Paragraph(
                        "Run key: E00 = BM25 latest; E01 = user history; E02 = rewrite; B02 = original + rewrite fusion; B03 = fusion + reranker; B04 = retrieval/reranker rank blend; R00 = latest + reranker. Full experiment identifiers and settings are saved in configs/ and runs/.",
                        styles["BodyText"],
                    )
                )
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(source) and source[i].strip().startswith("|"):
                block.append(source[i].strip())
                i += 1
            rows = [
                [c.strip() for c in s.strip("|").split("|")]
                for s in [block[0], *block[2:]]
            ]
            for row in rows[1:]:
                row[0] = short(row[0])
            cells = [
                [Paragraph(escape(c), styles["BodyText"]) for c in row] for row in rows
            ]
            n = len(rows[0])
            first = 130 if n == 3 else 90
            widths = [first] + [(495 - first) / (n - 1)] * (n - 1)
            tab = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            tab.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#8295a9")),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([tab, Spacer(1, 10)])
            continue
        paragraph = [line]
        i += 1
        while (
            i < len(source)
            and source[i].strip()
            and not source[i].startswith(("#", "|"))
        ):
            paragraph.append(source[i].strip())
            i += 1
        story.append(Paragraph(escape(" ".join(paragraph)), styles["BodyText"]))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            50, 28, "CS421 | Experiment report | Course-format review pending"
        )
        canvas.drawRightString(545, 28, str(doc.page))

    target = ROOT / "report/PROJECT_REPORT.pdf"
    SimpleDocTemplate(
        str(target),
        pagesize=(595, 842),
        leftMargin=50,
        rightMargin=50,
        topMargin=45,
        bottomMargin=45,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print("Exported:", target)


if __name__ == "__main__":
    main()
