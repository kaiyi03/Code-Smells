# Attribution

The prompt set used in this study is not ours. It is:

> Code Smell Detection Dataset. 2026. GitHub.
> <https://github.com/rosawoo/code-smell>

```bibtex
@misc{codesmell2026,
  title        = {Code Smell Detection Dataset},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/rosawoo/code-smell}}
}
```

476 Python prompts, each written to induce a named code smell, following the
methodology of *Green Prompting: Characterizing Prompt-driven Energy Costs of LLM
Inference* (arXiv:2503.10666), from which its `action_keywords` and
`expected_token_depth` fields derive.

Reuse here is with the authors' permission, on the condition that the work is
cited — which it is, in `paper/references.bib` as `codesmell2026` and on the
generated report page.

`prompts.jsonl` is produced by `fetch_prompts.py`, which downloads the dataset
from the repository above and reshapes it into this project's schema. The
`generations_*.jsonl` files are model output produced from those prompts.

**One observation for the dataset's authors**, recorded here because it affects how
the data can be analysed: `expected_token_depth` is a one-to-one relabelling of
`complexity` (basic→low 173, intermediate→medium 157, advanced→high 146, with no
exceptions). The two fields carry identical information, so `expected_token_depth`
cannot serve as an independent variable. Measured output tokens vary freely and are
used instead.
