"""Produce a vector PDF from the sections of the self-contained Reveal deck.

Uses the same rendered content and embedded SVGs. No browser process is required.
"""
import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).resolve().parent


def literal(value):
    return json.dumps(value, ensure_ascii=False)


def inline(node):
    if isinstance(node, NavigableString):
        return "#text(" + literal(str(node).replace("\n", " ")) + ")"
    content = "".join(inline(c) for c in node.children)
    if node.name in ("b", "strong"):
        return "#strong[" + content + "]"
    if node.name == "a":
        return "#link(" + literal(node["href"]) + ")[" + content + "]"
    return content


def table(element):
    rows = element.select("tr")
    widths = {3: "(26%,34%,40%)", 2: "(24%,76%)"}
    heading = element.select_one("th").get_text(strip=True)
    if heading == "Обозначение в легенде":
        widths[2] = "(33%,67%)"
    if heading == "Метод":
        widths[3] = "(44%,28%,28%)"
    if "ошибка" in heading:
        widths[3] = "(54%,23%,23%)"
    if "Верхний хвост" in heading:
        widths[3] = "(34%,33%,33%)"
    if "Серия" == heading:
        widths[3] = "(22%,26%,52%)"
    if not heading:
        widths[3] = "(21%,39%,40%)"
    cells = []
    for row in rows:
        for cell in row.find_all(["td","th"],recursive=False):
            value = inline(cell)
            if cell.name == "th":
                value = "#text(fill:navy,weight:700)[" + value + "]"
            cells.append("[" + value + "]")
        cells.append("table.hline(stroke: .6pt + silver)")
    n = len(rows[0].select("th"))
    return ("#block(width:100%)[#set text(size:23pt)\n"
            "#table(columns:" + widths[n] + ", inset:(x:10pt,y:6pt),stroke:none,"
            "fill:(x,y)=>if y==0 { rgb(\"f0f2f6\") }, " + ",\n".join(cells) + ")]")


def body(section, directory):
    result = []
    for el in section.select_one(".slide-body").find_all(recursive=False):
        classes = el.get("class", [])
        if "context" in classes:
            result.append("#block[#text(size:22pt,fill:muted)[" + inline(el) + "]]")
        elif "takeaway" in classes:
            result.append("#block(width:100%,stroke:(left:3pt+blue),inset:(left:14pt,y:2pt))"
                          "[#text(fill:navy,weight:600)[" + inline(el) + "]]")
        elif el.name == "table":
            result.append(table(el))
        elif el.select_one("img"):
            img = el.select_one("img")
            name = section["id"] + ".svg"
            data = img.get("src") or img["data-src"]
            (directory / name).write_bytes(base64.b64decode(data.partition(",")[2]))
            height = "290pt" if "architecture" in classes else "251pt"
            result.append("#align(center)[#image(" + literal(name) + f", width:100%,height:{height},fit:\"contain\")]")
        else:
            result.append("#block[" + inline(el) + "]")
    return "\n#v(11pt)\n".join(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--typst", default="typst", help="Typst binary (also bundled with Quarto)")
    args = ap.parse_args()
    html = BeautifulSoup((HERE / "fhcal.html").read_text(), "html.parser")
    sections = html.select("section.slide")
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for weight, style in ((400, "Regular"), (600, "Semibold"), (700, "Bold")):
            font = TTFont(HERE / "assets/SourceSans3VF-Upright.woff2")
            font = instantiateVariableFont(font, {"wght": weight}, inplace=True)
            font.flavor = None
            names = {1: "FHCal Slides", 2: style, 4: "FHCal Slides " + style,
                     6: "FHCalSlides-" + style, 16: "FHCal Slides", 17: style}
            for record in font["name"].names:
                if record.nameID in names:
                    record.string = names[record.nameID].encode(record.getEncoding())
            font.save(temp / (style + ".ttf"))
        source = ['''
#set document(title:"BM@N FHCal — условная генерация секционного отклика")
#set page(width:1200pt,height:675pt,margin:(x:58pt,y:28pt))
#set text(font:("FHCal Slides","Arial"),size:25pt,lang:"ru",fill:rgb("202125"))
#set par(leading:.37em,justify:false)
#let navy = rgb("0f2d69")
#let blue = rgb("374b9b")
#let silver = rgb("c7c8ca")
#let muted = rgb("4e5155")
#show link: set text(fill:blue)
#show link: underline
''']
        for i, sec in enumerate(sections):
            title = inline(sec.h2)
            footer = inline(sec.select_one(".slide-footnotes"))
            if i:
                source.append("#pagebreak()")
            source.append("#grid(columns:(1fr,),rows:(auto,1fr,auto),row-gutter:7pt,"
                          "[#text(size:34pt,weight:700,fill:navy)[" + title + "]"
                          "#v(7pt)\n#line(length:100%,stroke:1pt+silver)],"
                          "[#align(horizon)[" + body(sec, temp) + "]],"
                          "[#line(length:100%,stroke:.6pt+silver)\n"
                          "#grid(columns:(1fr,auto),column-gutter:12pt,"
                          "[#text(size:16pt,fill:muted)[" + footer + "]],"
                          "[#text(size:16pt,fill:muted)[" + str(i+1) + " / " + str(len(sections)) + "]])])")
        typ = temp / "slides.typ"
        typ.write_text("\n".join(source))
        subprocess.run([args.typst,"compile","--font-path",str(temp),str(typ),str(HERE/"fhcal.pdf")],check=True)


if __name__ == "__main__":
    main()
