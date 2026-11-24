"""Render reports/Final_Report.md to DOCX + PDF, split it into per-module documents, and build the slide deck.

Supported Markdown subset: #/##/### headings, paragraphs, **bold**, *italic*, `code`, bullet and numbered lists,
pipe tables, images ![caption](path), and '<!-- pagebreak -->'.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ACCENT = RGBColor(0x1F, 0x4E, 0x8C)

# ------------------------------------------------------------------------------------ parsing
def parse(md: str) -> list[tuple]:
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1; continue
        if ln.strip() == "<!-- pagebreak -->":
            blocks.append(("pagebreak",)); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip())); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            blocks.append(("img", m.group(1), m.group(2))); i += 1; continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows)); continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                items.append(item)
            blocks.append(("list", ordered, items)); continue
        para = [ln.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|!\[|\||\s*([-*]|\d+\.)\s|<!--)", lines[i].lstrip()):
            para.append(lines[i].strip()); i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`)")


def runs(text: str):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            yield part[2:-2], "b"
        elif part.startswith("`"):
            yield part[1:-1], "c"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            yield part[1:-1], "i"
        else:
            yield part, ""


# --------------------------------------------------------------------------------------- DOCX
def _shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def _add_runs(par, text, size=None):
    for t, style in runs(text):
        r = par.add_run(t)
        r.bold = style == "b"; r.italic = style == "i"
        if style == "c":
            r.font.name = "Consolas"
        if size:
            r.font.size = Pt(size)


def to_docx(blocks, out: Path, base: Path):
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2); sec.top_margin = sec.bottom_margin = Cm(2)
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for lvl, size in [(1, 18), (2, 14), (3, 12), (4, 11)]:
        h = doc.styles[f"Heading {lvl}"]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = ACCENT
    for b in blocks:
        if b[0] == "h":
            if b[1] == 1 and b is blocks[0]:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(b[2]); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = ACCENT
            else:
                doc.add_heading(re.sub(r"[*`]", "", b[2]), level=min(b[1], 4))
        elif b[0] == "p":
            p = doc.add_paragraph(); _add_runs(p, b[1]); p.paragraph_format.space_after = Pt(6)
        elif b[0] == "list":
            for it in b[2]:
                p = doc.add_paragraph(style="List Number" if b[1] else "List Bullet"); _add_runs(p, it)
        elif b[0] == "table":
            rows = b[1]; ncol = max(len(r) for r in rows)
            t = doc.add_table(rows=len(rows), cols=ncol); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, r in enumerate(rows):
                for j in range(ncol):
                    c = t.cell(i, j); c.text = ""
                    _add_runs(c.paragraphs[0], r[j] if j < len(r) else "", size=8.5)
                    if i == 0:
                        _shade(c, "DCE6F2")
                        for run in c.paragraphs[0].runs: run.bold = True
            doc.add_paragraph()
        elif b[0] == "img":
            p = (base / b[2]).resolve()
            if p.exists():
                doc.add_picture(str(p), width=Cm(16.0)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = cap.add_run(b[1]); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
        elif b[0] == "pagebreak":
            doc.add_page_break()
    doc.save(out)


# ---------------------------------------------------------------------------------------- PDF
def _rl(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = []
    for t, s in runs(text):
        out.append(f"<b>{t}</b>" if s == "b" else f"<i>{t}</i>" if s == "i" else f"<font face='Courier'>{t}</font>" if s == "c" else t)
    return "".join(out)


def to_pdf(blocks, out: Path, base: Path, title: str):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5, spaceAfter=0)
    cap = ParagraphStyle("cap", parent=body, fontSize=8, textColor=colors.HexColor("#52514e"), alignment=TA_CENTER, spaceAfter=10)
    hs = {1: ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor("#1f4e8c"), spaceBefore=6),
          2: ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=colors.HexColor("#1f4e8c")),
          3: ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11, textColor=colors.HexColor("#1f4e8c")),
          4: ParagraphStyle("h4", parent=ss["Heading4"], fontSize=10)}
    titlest = ParagraphStyle("t", parent=hs[1], fontSize=20, alignment=TA_CENTER, spaceAfter=12)
    story, W = [], A4[0] - 4 * cm
    for b in blocks:
        if b[0] == "h":
            story.append(Paragraph(_rl(b[2]), titlest if (b[1] == 1 and b is blocks[0]) else hs[min(b[1], 4)]))
        elif b[0] == "p":
            story.append(Paragraph(_rl(b[1]), body))
        elif b[0] == "list":
            story.append(ListFlowable([ListItem(Paragraph(_rl(it), body), leftIndent=12) for it in b[2]],
                                      bulletType="1" if b[1] else "bullet", start="1" if b[1] else None, leftIndent=16,
                                      bulletFontSize=9 if b[1] else 7, bulletFormat="%s." if b[1] else None))
        elif b[0] == "table":
            rows = b[1]; ncol = max(len(r) for r in rows)
            data = [[Paragraph(_rl(r[j] if j < len(r) else ""), small) for j in range(ncol)] for r in rows]
            from reportlab.pdfbase.pdfmetrics import stringWidth
            cells_by_col = [[re.sub(r"[*`]", "", r[j]) if j < len(r) else "" for r in rows] for j in range(ncol)]
            # minimum = widest single word (bold font, 7.5pt) + cell padding; the rest shared by content length
            mins = [max(stringWidth(w, "Helvetica-Bold", 7.5) for c in col for w in (c.split() or [""])) + 14 for col in cells_by_col]
            want = [min(max(len(c) for c in col), 48) for col in cells_by_col]
            spare = max(W - sum(mins), 0)
            widths = [mn + spare * wt / max(sum(want), 1) for mn, wt in zip(mins, want)]
            if sum(widths) > W:
                widths = [w * W / sum(widths) for w in widths]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8b7b2")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story += [t, Spacer(1, 8)]
        elif b[0] == "img":
            p = (base / b[2]).resolve()
            if p.exists():
                from PIL import Image as PI
                w, h = PI.open(p).size
                iw = min(W, 16.5 * cm); ih = iw * h / w
                if ih > 20 * cm:
                    ih = 20 * cm; iw = ih * w / h
                story.append(KeepTogether([Image(str(p), width=iw, height=ih), Paragraph(_rl(b[1]), cap)]))
        elif b[0] == "pagebreak":
            story.append(PageBreak())

    def footer(canvas, doc_):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(2 * cm, 1.2 * cm, title); canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=title, author="Business Analytics capstone").build(story, onFirstPage=footer, onLaterPages=footer)


def split_modules(md: str) -> dict[str, str]:
    """Each '## Module N ...' chapter becomes its own document."""
    parts = re.split(r"(?m)^(?=## Module \d)", md)
    out = {}
    for p in parts[1:]:
        title = p.splitlines()[0].lstrip("# ").strip()
        chapter = re.split(r"(?m)^## (?!Module)", p)[0]
        num = re.match(r"Module (\d+)", title).group(1)
        out[f"Module_{num}"] = "# " + title + "\n\n" + "\n".join(chapter.splitlines()[1:]).replace("\n### ", "\n## ").replace("\n#### ", "\n### ")
    return out


if __name__ == "__main__":
    src = REPORTS / "Final_Report.md"
    md = src.read_text(encoding="utf-8")
    title = "Learner Behavior Segmentation for an Online Learning Platform"
    blocks = parse(md)
    to_docx(blocks, REPORTS / "Final_Report.docx", REPORTS)
    to_pdf(blocks, REPORTS / "Final_Report.pdf", REPORTS, title)
    print("final report: docx + pdf")
    mod_dir = REPORTS / "modules"; mod_dir.mkdir(exist_ok=True)
    for name, text in split_modules(md).items():
        (mod_dir / f"{name}.md").write_text(text.replace("](figures/", "](../figures/"), encoding="utf-8")
        b = parse(text)
        to_docx(b, mod_dir / f"{name}.docx", REPORTS)
        to_pdf(b, mod_dir / f"{name}.pdf", REPORTS, title)
        print("module doc:", name)
    if "--no-slides" not in sys.argv:
        import build_slides
        build_slides.build()
