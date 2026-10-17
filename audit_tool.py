#!/usr/bin/env python3
"""
audit_tool.py — Local, blind labeling tool for the manual audit (the data never leave this computer).

    .venv/bin/python3 audit_tool.py          then open  http://127.0.0.1:8765  in your own browser

Shows one hospital course at a time with the sampled claims (and citation checks) of that course.  The judge's labels and the
condition that produced each claim are hidden.  Every click is saved immediately to results_private/audit_labels.json, so you
can stop at any time and continue later.  Stop the tool with Ctrl+C.
"""
import json
import os
import shutil
import time
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = Path(os.environ.get("AUDIT_ITEMS", HERE / "results_private" / "audit_items.json"))
LABELS = Path(os.environ.get("AUDIT_LABELS", HERE / "results_private" / "audit_labels.json"))
BACKUPS = LABELS.parent / "audit_backups"


def backup_labels(min_age_s=600):
    """Keep a timestamped copy of the labels at most every ten minutes (tiny files; never deleted automatically)."""
    try:
        if not LABELS.exists():
            return
        BACKUPS.mkdir(exist_ok=True)
        newest = max((f.stat().st_mtime for f in BACKUPS.glob("audit_labels-*.json")), default=0)
        if time.time() - newest >= min_age_s:
            shutil.copy2(LABELS, BACKUPS / time.strftime("audit_labels-%Y%m%d-%H%M%S.json"))
    except Exception:  # noqa: BLE001  (a failed backup must never block a save)
        pass
SHOWN = {"item_id", "kind", "claim", "cited", "excerpts"}   # everything else (condition, judge scores, strata) stays hidden
LOCK = threading.Lock()

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Thesis audit</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#6e6e73;--line:#d9d9de;--acc:#1f5fbf;--s:#1e8449;--p:#b9770e;--u:#c0392b;--q:#7f8c8d}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:16px}button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--acc)}.bar{flex:1;min-width:160px;height:8px;background:#eee;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px;max-width:1500px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
#course{white-space:pre-wrap;max-height:calc(100vh - 150px);overflow:auto;font-family:Georgia,serif;font-size:15.5px;line-height:1.6}
#course mark{background:#fff2a8}.item{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px;outline:none}
.item.focus{border-color:var(--acc);box-shadow:0 0 0 2px #1f5fbf22}.claim{font-size:16px;margin:0 0 8px}
.ex{background:#f3f4f8;border-radius:8px;padding:8px;margin:6px 0;font-size:14px;white-space:pre-wrap}
.btns{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}.btns button.on{color:#fff}
.b-S.on,.b-Yes.on{background:var(--s);border-color:var(--s)}.b-P.on,.b-Partly.on{background:var(--p);border-color:var(--p)}
.b-U.on,.b-No.on{background:var(--u);border-color:var(--u)}.b-Q.on{background:var(--q);border-color:var(--q)}
select,input[type=text]{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:8px}input[type=text]{width:100%;margin-top:6px}
.tag{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}.saved{color:var(--mute);font-size:13px}
details{margin-bottom:12px}summary{cursor:pointer;color:var(--acc)}.guide li{margin:4px 0}kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
#find{width:220px}
</style></head><body>
<header><b>Manual audit</b><span id="pos"></span><div class="bar"><i id="prog"></i></div><span id="cnt"></span>
<button onclick="go(-1)">&larr; Previous course</button><button onclick="go(1)">Next course &rarr;</button><button onclick="nextUnlabeled()">Next unlabeled</button>
<input id="find" type="text" placeholder="Highlight a word in the course" oninput="hl()"><span class="saved" id="saved"></span></header>
<main><section class="panel"><div class="tag">Hospital course</div><div id="course"></div></section>
<section class="panel"><details open><summary>How to label (click to hide)</summary><ul class="guide">
<li><b>S - Supported</b>: every fact in the statement is stated in the hospital course or follows directly from it. Plain-language paraphrase is fine.</li>
<li><b>P - Partly supported</b>: the statement has several facts; at least one is supported and at least one is not.</li>
<li><b>U - Not supported</b>: the main content is not in the course, or it contradicts the course.</li>
<li><b>? - Unsure</b>: you cannot decide from the course.</li>
<li>For P and U, pick what is wrong. Generic advice or follow-up that the course does not mention counts as not supported; choose "generic advice / follow-up not in the course" for it.</li>
<li><b>Citation checks</b>: does the quoted excerpt itself support the statement? Judge only the excerpt, not the whole course.</li>
<li>Keys: click a statement, then <kbd>1</kbd> S, <kbd>2</kbd> P, <kbd>3</kbd> U, <kbd>4</kbd> ?; <kbd>n</kbd> next course, <kbd>b</kbd> previous course. Everything saves automatically.</li></ul></details>
<div id="items"></div></section></main>
<script>
let D=null,L={},ci=0,focusId=null;
const TYPES=[["",""],["not_stated","not stated in the course"],["generic","generic advice / follow-up not in the course"],["contradicts","contradicts the course"],["med_number","wrong medication, dose or number"],["other","other"]];
async function load(){D=await (await fetch('data')).json();L=await (await fetch('labels')).json();
 try{const m=JSON.parse(localStorage.getItem('labels_mirror')||'{}');let rec=0;
  for(const k in m){if(m[k]&&m[k].label&&!(L[k]&&L[k].label)){L[k]=m[k];pend.add(k);rec++}}
  if(rec){flush();document.getElementById('saved').textContent='recovered '+rec+' unsaved answer'+(rec>1?'s':'')+' kept by this browser'}}catch(e){}
 const s=localStorage.getItem('ci');ci=s?Math.min(+s,D.courses.length-1):0;render()}
function esc(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function hl(){const q=document.getElementById('find').value.trim();const t=esc(D.courses[ci].text);if(!q){document.getElementById('course').innerHTML=t;return}
 const re=new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'gi');document.getElementById('course').innerHTML=t.replace(re,m=>'<mark>'+m+'</mark>')}
function total(){let n=0,d=0;for(const c of D.courses)for(const it of c.items){n++;if(L[it.item_id]&&L[it.item_id].label)d++}return[n,d]}
function render(){const c=D.courses[ci];localStorage.setItem('ci',ci);document.getElementById('pos').textContent=`Course ${ci+1} of ${D.courses.length}`;
 hl();document.getElementById('course').scrollTop=0;const [n,d]=total();document.getElementById('cnt').textContent=`${d} of ${n} labeled`;document.getElementById('prog').style.width=(100*d/n)+'%';
 const box=document.getElementById('items');box.innerHTML='';
 c.items.forEach((it,k)=>{const lab=L[it.item_id]||{};const div=document.createElement('div');div.className='item';div.tabIndex=0;div.dataset.id=it.item_id;
  div.onclick=()=>{focusId=it.item_id;document.querySelectorAll('.item').forEach(e=>e.classList.toggle('focus',e.dataset.id===focusId))};
  if(it.kind==='claim'){div.innerHTML=`<div class="tag">Statement ${k+1}</div><p class="claim">${esc(it.claim)}</p>`+
   `<div class="btns">${[['S','S - Supported'],['P','P - Partly'],['U','U - Not supported'],['Q','? - Unsure']].map(([v,t])=>`<button class="b-${v} ${lab.label===v?'on':''}" onclick="setL('${it.item_id}','${v}')">${t}</button>`).join('')}</div>`+
   `<select onchange="setT('${it.item_id}',this.value)" ${lab.label==='P'||lab.label==='U'?'':'style="display:none"'}>${TYPES.map(([v,t])=>`<option value="${v}" ${lab.type===v?'selected':''}>${t||'What is wrong?'}</option>`).join('')}</select>`+
   `<input type="text" placeholder="Optional note" value="${esc(lab.note||'')}" onchange="setN('${it.item_id}',this.value)">`}
  else{div.innerHTML=`<div class="tag">Citation check ${k+1}</div><p class="claim">${esc(it.claim)}</p>`+it.excerpts.map((e,j)=>`<div class="ex"><b>Cited excerpt ${it.cited[j]}</b>\n${esc(e)}</div>`).join('')+
   `<div class="btns">${[['Yes','Yes - supports'],['Partly','Partly'],['No','No'],['Q','? - Unsure']].map(([v,t])=>`<button class="b-${v} ${lab.label===v?'on':''}" onclick="setL('${it.item_id}','${v}')">${t}</button>`).join('')}</div>`+
   `<input type="text" placeholder="Optional note" value="${esc(lab.note||'')}" onchange="setN('${it.item_id}',this.value)">`}
  box.appendChild(div)})}
const pend=new Set();
function mirror(){try{localStorage.setItem('labels_mirror',JSON.stringify(L))}catch(e){}}
async function flush(){if(!pend.size)return;const body={};for(const k of pend)body[k]=L[k];let ok=false;
 try{ok=(await fetch('save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).ok}catch(e){ok=false}
 const el=document.getElementById('saved');
 if(ok){pend.clear();el.textContent='saved';el.style.cssText=''}
 else{el.textContent='NOT SAVING - the tool is not running. Your clicks are kept in this page; restart the tool in Terminal and they will save by themselves.';el.style.cssText='color:#b3261e;font-weight:700'}}
function save(id){mirror();pend.add(id);document.getElementById('saved').textContent='saving...';flush()}
setInterval(flush,5000);
function setL(id,v){L[id]=Object.assign(L[id]||{},{label:v});if(v!=='P'&&v!=='U')delete L[id].type;save(id);const y=window.scrollY;render();window.scrollTo(0,y);focusId=id;
 document.querySelectorAll('.item').forEach(e=>e.classList.toggle('focus',e.dataset.id===focusId))}
function setT(id,v){L[id]=Object.assign(L[id]||{},{type:v});save(id)}
function setN(id,v){L[id]=Object.assign(L[id]||{},{note:v});save(id)}
function go(d){ci=Math.max(0,Math.min(D.courses.length-1,ci+d));render();window.scrollTo(0,0)}
function nextUnlabeled(){for(let k=0;k<D.courses.length;k++){const j=(ci+1+k)%D.courses.length;if(D.courses[j].items.some(it=>!(L[it.item_id]&&L[it.item_id].label))){ci=j;render();window.scrollTo(0,0);return}}alert('Everything is labeled. Thank you!')}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
 if(e.key==='n')go(1);else if(e.key==='b')go(-1);else if(focusId){const it=D.courses[ci].items.find(x=>x.item_id===focusId);if(!it)return;
 const m=it.kind==='claim'?{'1':'S','2':'P','3':'U','4':'Q'}:{'1':'Yes','2':'Partly','3':'No','4':'Q'};if(m[e.key])setL(focusId,m[e.key])}});
load();
</script></body></html>"""


def _strip(payload):
    return dict(courses=[dict(sid=c["sid"], text=c["text"], items=[{k: v for k, v in it.items() if k in SHOWN} for it in c["items"]])
                         for c in payload["courses"]])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/data":
            self._send(200, json.dumps(_strip(json.load(open(ITEMS)))).encode(), "application/json")
        elif self.path == "/labels":
            self._send(200, (LABELS.read_bytes() if LABELS.exists() else b"{}"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, b"not found", "text/plain")
        upd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK:
            cur = json.loads(LABELS.read_text()) if LABELS.exists() else {}
            cur.update(upd)
            tmp = LABELS.with_suffix(".tmp"); tmp.write_text(json.dumps(cur, indent=1)); tmp.replace(LABELS)
            backup_labels()
        self._send(200, b"ok", "text/plain")


def main():
    if not ITEMS.exists():
        sys.exit("results_private/audit_items.json not found; run: .venv/bin/python3 audit_sample.py")
    port = int(os.environ.get("AUDIT_PORT", 8765))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    n = sum(len(c["items"]) for c in json.load(open(ITEMS))["courses"])
    done = len([v for v in (json.loads(LABELS.read_text()) if LABELS.exists() else {}).values() if v.get("label")])
    print(f"Audit tool running at http://127.0.0.1:{port}  ({done} of {n} items labeled so far). Stop with Ctrl+C.", flush=True)
    print(f"Every click saves to {LABELS}; a timestamped copy is kept in {BACKUPS}/ at most every 10 minutes.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; labels are saved in results_private/audit_labels.json")


if __name__ == "__main__":
    main()
