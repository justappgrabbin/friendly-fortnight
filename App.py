from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import git, os, json, ast, subprocess, tempfile, faiss, numpy as np

app = Flask(__name__)

# load models once
tok = AutoTokenizer.from_pretrained("./models/codet5")
mdl = AutoModelForSeq2SeqLM.from_pretrained("./models/codet5")
emb = SentenceTransformer("./models/embedding")

INDEX_FILE = "code.index"
META_FILE  = "code_meta.json"

def clone_and_index(repo_url):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    if not os.path.exists("repos"): os.makedirs("repos")
    repo_path = os.path.join("repos", repo_name)
    if not os.path.exists(repo_path):
        git.Repo.clone_from(repo_url, repo_path)
    chunks = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f)) as fp:
                    txt = fp.read()
                # simple 20-line sliding window
                lines = txt.splitlines()
                for i in range(0, len(lines), 20):
                    chunk = "\n".join(lines[i:i+20])
                    chunks.append({"code": chunk, "file": os.path.join(root, f), "start": i})
    if not chunks: return
    codes = [c["code"] for c in chunks]
    vectors = emb.encode(codes)
    idx = faiss.IndexFlatL2(vectors.shape[1])
    idx.add(vectors.astype(np.float32))
    faiss.write_index(idx, INDEX_FILE)
    with open(META_FILE, "w") as fp:
        json.dump(chunks, fp)
    return chunks

def rag_retrieve(query, k=3):
    if not os.path.exists(INDEX_FILE): return []
    idx = faiss.read_index(INDEX_FILE)
    with open(META_FILE) as fp:
        meta = json.load(fp)
    q = emb.encode([query])
    _, I = idx.search(q, k)
    return [meta[i] for i in I[0] if i < len(meta)]

@app.route("/index_repo", methods=["POST"])
def index_repo():
    url = request.json["repo_url"]
    try:
        clone_and_index(url)
        return jsonify({"message": "Repository indexed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/generate", methods=["POST"])
def generate():
    code = request.json["code"]
    snippets = rag_retrieve(code)
    context = "\n".join([s["code"] for s in snippets])
    prompt = (f"Using these examples:\n{context}\n\nGenerate Python code for:\n{code}"
              if context else code)[:512]
    ids = tok(prompt, return_tensors="pt").input_ids
    out = mdl.generate(ids, max_length=256)
    return jsonify({"generated_code": tok.decode(out[0], skip_special_tokens=True)})

@app.route("/fix", methods=["POST"])
def fix():
    code = request.json["code"]
    snippets = rag_retrieve(code)
    context = "\n".join([s["code"] for s in snippets])
    prompt = (f"Using these examples:\n{context}\n\nFix and complete:\n{code}"
              if context else f"Fix and complete:\n{code}")[:512]
    ids = tok(prompt, return_tensors="pt").input_ids
    out = mdl.generate(ids, max_length=256)
    return jsonify({"fixed_code": tok.decode(out[0], skip_special_tokens=True)})

INDEX_HTML = '''
<!doctype html>
<title>Universal RAG Code Tool</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;padding:2rem}
textarea{width:100%;height:300px;background:#222;color:#0f0}
button{background:#0f0;color:#111;border:none;padding:.6rem 1.2rem;margin:.3rem}
</style>
<h1>🧠 Universal RAG Code Tool</h1>
<input id="url" placeholder="Any GitHub URL" size="60" value="https://github.com/AmirAbaskohi/PythonCodeGenerator.git">
<button onclick="index()">Index Repo</button><br><br>
<textarea id="code" placeholder="Paste code or prompt"></textarea><br>
<button onclick="gen()">Generate</button>
<button onclick="fix()">Fix / Complete</button>
<pre id="out"></pre>
<script>
async function index(){
  const u=document.getElementById("url").value;
  const r=await fetch("/index_repo",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({repo_url:u})});
  const j=await r.json(); alert(j.message||j.error);
}
async function gen(){
  const c=document.getElementById("code").value;
  const r=await fetch("/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({code:c})});
  const j=await r.json(); document.getElementById("out").textContent=j.generated_code;
}
async function fix(){
  const c=document.getElementById("code").value;
  const r=await fetch("/fix",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({code:c})});
  const j=await r.json(); document.getElementById("out").textContent=j.fixed_code;
}
</script>
'''

@app.route("/")
def home():
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
