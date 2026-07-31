# The paper

`main.tex` is the draft; `references.bib` is the bibliography. The introduction,
abstract, contributions and research questions are written. Every other section
carries a comment block listing what belongs in it and which results it draws on,
so the sections can be written in any order.

## Editing it on Overleaf

Overleaf runs LaTeX in the browser, needs no install, and makes it easy to share
a link for comments.

1. Go to [overleaf.com](https://www.overleaf.com) and sign in — Oxford has
   institutional access via **Log in through your institution**.
2. **New Project → Upload Project**, and upload `main.tex` and `references.bib`
   (select both, or zip the `paper/` folder and upload the zip).
3. Press **Recompile**. Overleaf runs BibTeX for you.
4. **Share → Turn on link sharing** to send a review link.

Overleaf becomes the working copy once it's up. To keep the repo in sync,
download the `.tex` from Overleaf and commit it, or connect Overleaf to GitHub
from the same Share menu.

## Compiling locally instead

Needs a TeX distribution (MiKTeX on Windows, TeX Live elsewhere):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Four passes: the first resolves the text, BibTeX builds the bibliography, and the
last two settle the citations and cross-references.
