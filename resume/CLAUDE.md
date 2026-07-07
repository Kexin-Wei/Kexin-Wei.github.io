# Resume — build notes

Three resumes, same toolchain:

- `academic/resume.tex` — academic CV, comprehensive; shows all background but
  foregrounds robotics & AI (Research Interests section + accent-coloured
  `\rai{...}` labels). Uses `biblatex` + `Publications.bib`.
- `job/resume-aiml.tex` — one-page, AI/ML-targeted industry resume.
- `job/resume-robotics.tex` — one-page, Robotics/Medical-targeted resume.

## job/ two-variant architecture

`resume-aiml.tex` and `resume-robotics.tex` are 2-line drivers that set
`\def\resumeversion{aiml|robotics}` then `\input{resume-body.tex}`. All shared
content lives in **`resume-body.tex`**; edit it once and both PDFs update.
Version-specific content/style is guarded by `\ifaimlver ... \else ... \fi`
(tagline, which Experience bullets show, skills ordering). Accent colour
(`UI_blue`) is intentionally **identical** across all three docs for brand
consistency.

## Metric placeholders

Both `resume-body.tex` and `academic/resume.tex` define `\ph{...}`, which renders
a **bright red `[...]`** marker. Use it for numbers you don't have yet (never
fabricate metrics on a CV); the red makes an unfilled figure impossible to ship
by accident. Replace with the real value to make it disappear.

## Building

All documents use `fontspec` with custom OTF fonts (`SourceSansPro-*.otf`), so
they **must** be compiled with **XeLaTeX** — `pdflatex` fails with
`fontspec requires XeTeX or LuaTeX`.

Each folder has a `latexmkrc` (`$pdf_mode = 5`) so plain `latexmk` picks XeLaTeX
automatically. `job/latexmkrc` also sets `@default_files` to the two drivers, so
bare `latexmk` builds both variants and ignores `resume-body.tex`. Build inside
the `sanjibsen/weblatex` Docker container:

```bash
# academic/
latexmk resume.tex
# job/  (builds resume-aiml.pdf and resume-robotics.pdf)
latexmk
```

## Gotchas

- A crashed `pdflatex` run leaves stale `.aux`/`.fls`/`.log` that poison the
  next `latexmk`. Always `latexmk -C` first if a build misbehaves.
- FontAwesome icons emit harmless `ToUnicode CMap failed` warnings — ignore.
- `fancyhdr` warns `\footskip is too small` — cosmetic, ignore.
