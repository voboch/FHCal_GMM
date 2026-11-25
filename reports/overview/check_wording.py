"""Check for known wording regressions before rendering the overview."""
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
UNDEFINED_SAMPLE = re.compile(
    r"\b(?:мал\w*|больш\w*)\s+(?:(?:исходн|условн)\w*\s+)?банк\w*\b", re.I
)
EMPTY_OPENING = re.compile(
    r"\b(?:важно отметить|стоит отметить|следует отметить|it's worth noting)\b", re.I
)
CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def check(source):
    messages = []
    draft = source.read_text()
    for line_no, line in enumerate(draft.splitlines(), 1):
        for pattern in (UNDEFINED_SAMPLE, EMPTY_OPENING):
            if match := pattern.search(line):
                messages.append(f"{source.name}:{line_no}: replace '{match[0]}'")
    for filename in re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", draft):
        # Matplotlib stores labels in comments when it outlines the glyphs.
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        for node in ET.parse(source.parent / filename, parser).iter():
            if node.tag is ET.Comment or str(node.tag).split("}")[-1] in {"text", "tspan", "title", "desc"}:
                label = "".join(node.itertext())
                if CYRILLIC.search(label):
                    messages.append(f"{filename}: use an English figure label for '{label}'")
    return messages


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fhcal.qmd"
    errors = check(source)
    if errors:
        sys.exit("\n".join(errors))
