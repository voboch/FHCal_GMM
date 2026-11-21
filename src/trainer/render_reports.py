"""Build the three scientific PDFs from their Markdown and vector plots."""

import argparse
import os
from pathlib import Path
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "src/gaussian_projection"
PROTON = PILOT / "proton_pilot"
REPORTS = (
    (PILOT / "RESULTS.md", "FHCal_Gaussian_Mixture_Pilot.pdf", ("## Энергия мюонов в модулях",)),
    (PROTON / "RESULTS.md", "FHCal_Proton_Projection_Pilot.pdf",
     ("### Как учли остановки и что дал лимит 800", "## При увеличении числа компонент с 4 до 8",
      "## Ошибка по событиям связана прежде всего с занятостью", "## Сравнение пяти записанных направлений",
      "## Полный сигнал и отклик отдельных модулей")),
    (PROTON / "conditional_baseline/DECODER_ANALYSIS.md", "FHCal_Stochastic_Decoder_Report.pdf",
     ("## Продольное соседство", "## Что меняется", "## Полный сигнал и правый хвост модуля",
      "## Вес поглотителя в продольной координате")),
)


def printable_markdown(source, destination, breaks):
    def move_link(match):
        href = match[1]
        if "://" in href or href.startswith("#"):
            return match[0]
        name, marker, anchor = href.partition("#")
        path = (source.parent / name).resolve(strict=True)
        if path.parent.name in {"figures", "decoder_figures"} and path.suffix == ".png":
            path = path.with_suffix(".pdf").resolve(strict=True)
        return "](" + os.path.relpath(path, destination) + marker + anchor + ")"

    markdown = re.sub(r"\]\(([^\s)]+)\)", move_link, source.read_text(encoding="utf-8"))
    for heading in breaks:
        marker = "\n" + heading
        if markdown.count(marker) != 1:
            raise ValueError(f"Page break does not match a unique heading in {source.name}: {heading}")
        markdown = markdown.replace(marker, "\n\\newpage\n\n" + heading)
    return markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarto", default="quarto")
    parser.add_argument("--only", choices=[filename for _, filename, _ in REPORTS])
    args = parser.parse_args()
    reports = [entry for entry in REPORTS if args.only is None or entry[1] == args.only]
    destination = ROOT / "reports"
    environment = os.environ.copy()
    environment.setdefault("SOURCE_DATE_EPOCH", "1795132800")
    environment.setdefault("FORCE_SOURCE_DATE", "1")
    with TemporaryDirectory(prefix="fhcal-pdf-") as temporary:
        stage = Path(temporary)
        for source, filename, breaks in reports:
            transformed = (stage / filename).with_suffix(".md")
            transformed.write_text(printable_markdown(source, destination, breaks), encoding="utf-8")
            subprocess.run(
                [args.quarto, "pandoc", "--fail-if-warnings", "--from=markdown-implicit_figures", "--pdf-engine=xelatex",
                 "--standalone", "--metadata", "lang=ru", "-V", "papersize=a4", "-V", "geometry:top=14mm",
                 "-V", "geometry:bottom=14mm", "-V", "geometry:left=18mm", "-V", "geometry:right=18mm",
                 "-V", "mainfont=Arial", "-V", "monofont=Courier New", "-V", "fontsize=10pt", "-V", "colorlinks=true",
                 "--resource-path=.", "--output", str(stage / filename), str(transformed)],
                cwd=destination, env=environment, check=True,
            )
        for _, filename, _ in reports:
            copyfile(stage / filename, destination / filename)


if __name__ == "__main__":
    main()
