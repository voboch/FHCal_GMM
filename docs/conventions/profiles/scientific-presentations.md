# Scientific presentations and thesis reuse
User instruction, 2026-11-16. Applies to scientific presentations in Thesis
and adopting projects, including status, pre-defense and defense decks.
Use the reviewed `quarto-reveal-teaching-kit` as the layout reference.
Explicit user instructions for a particular artifact take precedence.
The requirements for scientific reporting and figures are in
[§25](../25-scientific-reporting.md) and [§20](../20-result-representation.md).
The reviewed source is the user's
[quarto-reveal-teaching-kit.zip](https://chatgpt.com/library?file_id=libfile_fde3_f-73867f55eec08191995730f2e2500a07_55231900d3188191ac151d12e0442379&account_id=personal).

## 1. Local Quarto is the presentation workflow

- Author editable `.qmd` sources and run Quarto/Reveal.js locally. Keep the
  configuration, theme and required assets with the source. Use local Quarto
  rendering instead of automatically routing the task to a cloud slide editor
  or a different presentation generator.
- At the first build, check `quarto --version` and the actual Python, Node and
  browser dependencies used. Reuse the project's working environment and record
  versions in the existing build/review record. If a required component cannot
  run, identify the unverified stage; do not claim that a build or check passed.
- Run `quarto render <presentation-directory>` with the actual project path.
  Export the resulting HTML to PDF locally using the kit's `export-pdf.cjs` or
  a verified equivalent. The kit script accepts input HTML and output PDF paths.
- Keep fonts, MathJax and required resources local or embedded. Deliver HTML
  for presenting and PDF for distribution from the same source revision;
  preserve the editable source and its assets. Test the delivered HTML offline.
  External source links may require a connection.

## 2. Reuse a small, stable layout system

Reuse `theme.scss`, `slide-layout.scss` and `slide-body.lua`, plus local fonts,
MathJax and the export workflow. Separate general classes from `lecture.scss`
and make lecture-specific CSS, data and JavaScript optional. A new deck must
not load scripts that require sliders or elements absent from that deck.

The baseline is 1600 × 900 with Source Sans 3, 36 px body, 46 px working-slide
headings and 23 px sources. Changing the canvas requires updating repeated
dimensions in the styles. Check the actual display size; these defaults do not
justify unreadable graph labels. Preserve the scientific plot grammar and
reference/prior/model encodings when applying colors and typography. See
[§20 F4](../20-result-representation.md#f4-график-показывает-доказательство)
and [§20 F7–F8](../20-result-representation.md#f7-профиль-презентации).

Use the HSE University palette for Quarto slide accents. Use navy `#0F2D69`
for headings and cover accents, blue `#374B9B` for links and callout rules,
and the brand grays `#939598`, `#C7C8CA`, and `#E6E7E8` for rules and table
backgrounds. Preserve the scientific encodings in plots. See the
[HSE brandbook](https://www.hse.ru/info/brandbook/) and
[official HTML palette](https://portal.hse.ru/colours).

Use `##` slide headings, at most one top-level `.slide-footnotes` block and
top-level `.notes`, as expected by the kit filter. Keep title, content, sources
and speaker notes distinct. Use only the layouts needed by the argument:
status summary; problem/data; mechanism; main result; controlled comparison;
robustness/limitation; next decision. No fixed slide count or panel count is required.

## 3. Choose the content profile

| Material | Main sequence |
|---|---|
| Status | What changed since the previous report; decisive comparison and evidence; current limitation; next decision and the check that will inform it |
| Defense / pre-defense | Research problem and tasks; original contribution; claims and their evidence; scope and limitations; conclusions |
| Thesis text | Connected argument with complete captions and chapter conclusions, using the existing thesis authoring format and page layout |

For a status, bring back definitions and method details when they are needed to
assess the new evidence. For a defense, distinguish the author's contribution
from referenced methods. Put supporting derivations and exhaustive breakdowns
in an appendix when they do not determine the main conclusion. A material
failure or limitation remains in the main argument.

For a status deck, lead with the scientific comparison. On the first
comparison slide, name the methods and reference, then give the shared
sample and operating conditions before the result. Introduce seed or
fold details where they explain variability. Keep the reason for a
limitation visible when it changes what can be inferred. For example,
entrance crossings define which neutrons reached the detector and ancestry
links their showers to those neutrons.

## 4. One result source for all representations

Consume verified canonical exports from the research repositories. Reuse data,
definitions, notation, source mappings and figure-generation code across
slides and thesis text. Use separate display/print settings for figure size,
labels, line widths and panels. Use the existing manuscript format and the
[§20 F6 print profile](../20-result-representation.md#f6-профиль-статьи-и-диссертации)
for thesis figures.

The kit's `figures/build.py` generates illustrative data; its CSVs are not
experimental evidence. Adapt its rendering helpers to read project exports.
Generate tables, chart values and numeric text inserts from the same source.
Check captions, alt text and prose after changes so an updated plot cannot keep
an obsolete explanation. Label illustrations visibly and identify the actual
population, comparison, uncertainty and scope of measured results. Follow
[§20 F1–F3](../20-result-representation.md#f1-число-имеет-однозначный-смысл)
for results and [§20 F5](../20-result-representation.md#f5-видимая-подпись-и-точный-источник)
for captions and source links.

Each result slide has one function and enough visible context to interpret its
evidence. Give the main graph adequate space, state the supported consequence
and show a short source reference. Put detailed run IDs, paths and hashes in
notes/source mappings. An interaction needs a PDF table or fixed states that
preserve the conclusion and use the same data.

When a result depends on matching, show what establishes a match and
when any uniqueness requirement applies. When it depends on a threshold,
state the origin of the operating point and the data role used to set it.
Reserve speaker notes for details the audience can skip while reading the
result correctly. Place any additional metric alongside the result it
qualifies, and state its own population, denominator and truth or
reconstructed binning.

## 5. Focused acceptance

Read the titles in order and check the argument without notes. Verify claims
against their sources. Inspect final HTML at audience size and the PDF,
including appendix pages: fonts, formulas, sources, axes, legends, clipping,
overlap, page boundaries and static replacements for interactive content.
Exercise interactive controls and keyboard behavior when present; check the
HTML offline. Automated export checks supplement this visual review.

Repair content or layout before reducing type size. Reuse accepted scientific
evidence for unchanged results. Finish after resolving material accuracy and
readability issues. Record any check that the available environment could not
complete. Follow [§25.2](../25-scientific-reporting.md#2-короткий-процесс)
for review scope and [§25.3](../25-scientific-reporting.md#3-замечание-и-вердикт)
for the verdict. New calculations follow [§10](../10-methodology.md)
and [§30](../30-development-process.md#8-review-gates).
