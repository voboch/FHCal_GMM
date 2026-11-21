"""Join the scientific reports, keeping their annotations and adding explicit bookmarks."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject

from render_reports import ROOT


SOURCES = (
    ("FHCal_Gaussian_Mixture_Pilot.pdf", "Мюонный пилот и энергия модулей"),
    ("FHCal_Proton_Projection_Pilot.pdf", "Протонный пилот и энергия модулей"),
    ("FHCal_Stochastic_Decoder_Report.pdf", "Стохастический декодер"),
)
MATERIAL = "Вес поглотителя в продольной координате"


def without_local_links(book):
    """Keep web and internal navigation actions in the copy for online reading."""
    for page in book.pages:
        annotations = page.get("/Annots", [])
        kept = ArrayObject()
        for reference in annotations:
            action = reference.get_object().get("/A")
            if action is not None:
                action = action.get_object()
                kind = action.get("/S")
                if kind in {"/GoToR", "/Launch"}:
                    continue
                if kind == "/URI" and urlsplit(str(action.get("/URI", ""))).scheme not in {"http", "https"}:
                    continue
            kept.append(reference)
        if annotations:
            if kept:
                page[NameObject("/Annots")] = kept
            else:
                del page["/Annots"]


def bundle(directory, destination, *, portable=False):
    book = PdfWriter()
    for name, title in SOURCES:
        reader = PdfReader(directory / name)
        first = len(book.pages)
        book.append(reader, import_outline=False)
        parent = book.add_outline_item(title, first)
        if name == SOURCES[-1][0]:
            material, = [i for i, page in enumerate(reader.pages) if page.extract_text().startswith(MATERIAL)]
            book.add_outline_item(MATERIAL, first + material, parent=parent)
    if portable:
        without_local_links(book)
    book.add_metadata({"/Title": "BM@N FHCal — пилоты, энергия модулей и продольная сегментация"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".pdf", delete=False) as temporary:
        stage = Path(temporary.name)
    try:
        with stage.open("wb") as temporary:
            book.write(temporary)
        stage.replace(destination)
    finally:
        stage.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-output", type=Path, help="Write a copy with web links and bookmarks only")
    args = parser.parse_args()
    directory = ROOT / "reports"
    destination = args.portable_output or directory / "FHCal_Reports_Current.pdf"
    bundle(directory, destination, portable=args.portable_output is not None)


if __name__ == "__main__":
    main()
