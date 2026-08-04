"""
One page showing every generation from every model (generations.html).

Both task sets in one place, grouped by task so a model's answer sits next to the
others' for the same prompt. Filter by task set or model, or search by task id or
by anything in the code.

Each pane is its own <pre> with its own copy button, so selecting one model's code
does not drag in the neighbouring column -- the same problem the sample browser had.

Run:  python view_generations.py
"""

import html
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generations.html")

# (file, task set, model label, short key). Order decides column order.
SOURCES = [
    ("arc_qwen/generations.jsonl",                 "benchmark", "Qwen2.5-Coder-1.5B",  "qwen"),
    ("arc_qwen/generations_deepseek.jsonl",        "benchmark", "DeepSeek-Coder-1.3B", "deepseek"),
    ("prompt_study/generations_qwen4k.jsonl",      "prompts",   "Qwen2.5-Coder-1.5B",  "qwen"),
    ("prompt_study/generations_deepseek4k.jsonl",  "prompts",   "DeepSeek-Coder-1.3B", "deepseek"),
    ("prompt_study/generations_claude.jsonl",      "prompts",   "Claude Sonnet 5",     "claude"),
]
SETS = {"benchmark": "MBPP + HumanEval", "prompts": "Smell-inducing prompts"}


def load():
    """task_id -> {"set":..., "prompt":..., "canonical":..., "by_model": {key: row}}"""
    tasks = defaultdict(lambda: {"by_model": {}})
    models, counts = {}, Counter()
    for rel, tset, label, key in SOURCES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"  [skip] {rel} not found")
            continue
        models[key] = label
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            t = tasks[(tset, r["task_id"])]
            t["set"] = tset
            t["task_id"] = r["task_id"]
            t["prompt"] = r.get("prompt") or r.get("instruction") or ""
            t["canonical"] = r.get("canonical_code") or ""
            t["intended"] = r.get("intended_tracked") or []
            t["by_model"][key] = r
            counts[(tset, key)] += 1
    return tasks, models, counts


CSS = """
:root{--ink:#16202b;--paper:#fff;--muted:#5b6b7a;--rule:#e2e8ee;--sur:#f6f8fa;
      --accent:#0f6d6d;--chip:#eef4f4}
@media(prefers-color-scheme:dark){:root{--ink:#e4e9ef;--paper:#11151a;--muted:#93a2b1;
      --rule:#28313b;--sur:#191e25;--accent:#4fb8b0;--chip:#16292a}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);line-height:1.55;
     font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:15px}
.wrap{max-width:1400px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:25px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px;font-size:14.5px;max-width:78ch}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--rule);
     padding:12px 0;margin-bottom:16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.bar input[type=search]{flex:1;min-width:220px;padding:7px 11px;font:inherit;font-size:14px;
     border:1px solid var(--rule);border-radius:7px;background:var(--sur);color:var(--ink)}
.bar label{font-size:13.5px;display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.count{color:var(--muted);font-size:13px;white-space:nowrap}
.card{border:1px solid var(--rule);border-radius:9px;margin:0 0 14px;overflow:hidden}
.hd{background:var(--sur);padding:8px 12px;display:flex;gap:10px;align-items:center;
    flex-wrap:wrap;font-size:13.5px}
.hd b{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.chip{background:var(--chip);color:var(--accent);border-radius:20px;padding:1px 9px;
      font-size:11.5px;font-weight:650}
.ask{padding:9px 12px;color:var(--muted);font-size:13.5px;border-bottom:1px solid var(--rule);
     max-width:110ch}
.cols{display:flex;gap:0;align-items:stretch;overflow-x:auto}
.col{flex:1 1 0;min-width:290px;border-right:1px solid var(--rule);display:flex;flex-direction:column}
.col:last-child{border-right:0}
.colhd{padding:6px 11px;font-size:12px;font-weight:650;color:var(--muted);
       border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;gap:8px}
pre{margin:0;padding:10px 12px;font-family:ui-monospace,Consolas,monospace;font-size:12.3px;
    line-height:1.45;overflow-x:auto;white-space:pre;flex:1}
button.copy{font:inherit;font-size:11px;padding:1px 8px;cursor:pointer;border:1px solid var(--rule);
     border-radius:5px;background:var(--paper);color:var(--muted)}
button.copy:hover{border-color:var(--accent);color:var(--accent)}
button.copy.done{border-color:#1b6b41;color:#1b6b41}
.hide{display:none}
.more{display:block;width:100%;padding:9px;margin:10px 0 30px;font:inherit;cursor:pointer;
      border:1px solid var(--rule);border-radius:8px;background:var(--sur);color:var(--ink)}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
let shown=60;
function apply(){
  const q=document.getElementById('q').value.toLowerCase().trim();
  const sets=[...document.querySelectorAll('.fset:checked')].map(c=>c.value);
  const mods=[...document.querySelectorAll('.fmod:checked')].map(c=>c.value);
  let n=0,vis=0;
  for(const c of cards){
    const okSet=sets.includes(c.dataset.set);
    const okQ=!q||c.dataset.hay.includes(q);
    const ok=okSet&&okQ;
    if(ok){n++;c.classList.toggle('hide',n>shown);if(n<=shown)vis++;}
    else c.classList.add('hide');
    if(ok) for(const col of c.querySelectorAll('.col'))
      col.classList.toggle('hide',col.dataset.model&&!mods.includes(col.dataset.model));
  }
  document.getElementById('count').textContent=
    n+' task'+(n===1?'':'s')+' match'+(n===1?'es':'')+(n>shown?' — showing '+shown:'');
  document.getElementById('more').style.display=n>shown?'block':'none';
}
document.getElementById('q').addEventListener('input',()=>{shown=60;apply()});
document.querySelectorAll('.fset,.fmod').forEach(c=>c.addEventListener('change',apply));
document.getElementById('more').addEventListener('click',()=>{shown+=60;apply()});
document.addEventListener('click',e=>{
  const b=e.target.closest('button.copy'); if(!b)return;
  const t=b.closest('.col').querySelector('pre').textContent;
  const done=()=>{const o=b.textContent;b.textContent='copied';b.classList.add('done');
    setTimeout(()=>{b.textContent=o;b.classList.remove('done')},1100)};
  if(navigator.clipboard&&navigator.clipboard.writeText)
    navigator.clipboard.writeText(t).then(done,()=>fb(t,done)); else fb(t,done);
});
function fb(t,done){const a=document.createElement('textarea');a.value=t;
  a.style.position='fixed';a.style.opacity='0';document.body.appendChild(a);a.select();
  try{document.execCommand('copy');done()}catch(e){}document.body.removeChild(a)}
apply();
"""


def main():
    tasks, models, counts = load()
    if not tasks:
        raise SystemExit("no generation files found")

    order = sorted(tasks.values(), key=lambda t: (t["set"] != "benchmark", t["task_id"]))
    parts = []
    for t in order:
        hay = (t["task_id"] + " " + t["prompt"]).lower()
        for r in t["by_model"].values():
            hay += " " + r["generated_code"][:600].lower()

        cols = []
        if t["canonical"]:
            cols.append(
                '<div class="col" data-model="canonical"><div class="colhd">'
                '<span>canonical solution</span>'
                '<button class="copy">copy</button></div>'
                f'<pre>{html.escape(t["canonical"])}</pre></div>')
        for key, label in models.items():
            r = t["by_model"].get(key)
            if not r:
                continue
            n = r.get("n_output_tokens")
            cols.append(
                f'<div class="col" data-model="{key}"><div class="colhd">'
                f'<span>{html.escape(label)}</span>'
                f'<span>{n} tok <button class="copy">copy</button></span></div>'
                f'<pre>{html.escape(r["generated_code"])}</pre></div>')

        tag = "".join(f'<span class="chip">{html.escape(s)}</span>' for s in t["intended"])
        ask = (f'<div class="ask">{html.escape(t["prompt"][:400])}'
               f'{"&hellip;" if len(t["prompt"]) > 400 else ""}</div>') if t["prompt"] else ""
        parts.append(
            f'<div class="card" data-set="{t["set"]}" data-hay="{html.escape(hay)}">'
            f'<div class="hd"><b>{html.escape(t["task_id"])}</b>'
            f'<span class="chip">{html.escape(SETS[t["set"]])}</span>{tag}</div>'
            f'{ask}<div class="cols">{"".join(cols)}</div></div>')

    setboxes = "".join(
        f'<label><input type="checkbox" class="fset" value="{k}" checked> {v}</label>'
        for k, v in SETS.items())
    modboxes = "".join(
        f'<label><input type="checkbox" class="fmod" value="{k}" checked> {v}</label>'
        for k, v in models.items())
    total = sum(counts.values())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generations &mdash; every model</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Generations</h1>
<p class="sub">Every solution produced by every model, grouped by task so the answers to
the same prompt sit side by side. {total:,} generations over {len(order):,} tasks.
Each pane copies on its own &mdash; selecting one column will not drag in its neighbour.</p>
<div class="bar">
  <input type="search" id="q" placeholder="search task id, prompt, or code&hellip;">
  <span class="count" id="count"></span>
</div>
<div class="bar" style="position:static;border:0;padding-top:0">
  <span class="count">task set:</span>{setboxes}
  <span class="count" style="margin-left:14px">models:</span>{modboxes}
</div>
{"".join(parts)}
<button class="more" id="more">Show more</button>
<script>{JS}</script>
</div></body></html>""")
    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {os.path.basename(OUT)}  ({total:,} generations, {len(order):,} tasks, "
          f"{size:.1f} MB)")
    for (tset, key), n in sorted(counts.items()):
        print(f"  {SETS[tset]:26} {models[key]:22} {n:5}")


if __name__ == "__main__":
    main()
