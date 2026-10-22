"""Pagewise web app: upload any text PDF, ask questions, get answers with page citations and the line highlighted."""
import os, re, json
from flask import Flask, request, jsonify, send_from_directory, Response, abort
from . import engine as E

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
ID = re.compile(r"^[0-9a-f]{10}$")
def need(i):
    if not ID.match(i) or not os.path.isdir(E.doc_dir(i)): abort(404)

@app.get("/")
def home(): return send_from_directory(app.static_folder, "index.html")
@app.get("/api/status")
def status(): return jsonify(has_key=bool(os.environ.get("GROQ_API_KEY")), model=E.R.MODEL)
@app.get("/api/docs")
def docs(): return jsonify(E.list_docs())
@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename: return jsonify(error="Choose a PDF file."), 400
    raw = f.read()
    if raw[:5] != b"%PDF-": return jsonify(error="That file is not a PDF."), 400
    return jsonify(id=E.add_document(raw, os.path.basename(f.filename)))
@app.delete("/api/doc/<i>")
def delete(i): need(i); return jsonify(ok=E.delete_doc(i))
@app.get("/api/doc/<i>")
def doc(i):
    need(i); m = E.read_json(os.path.join(E.doc_dir(i), "meta.json"), {}); m["chat"] = E.read_json(os.path.join(E.doc_dir(i), "chat.json"), []); return jsonify(m)
@app.get("/api/doc/<i>/page/<int:n>.png")
def page(i, n):
    need(i); b = E.render_page(i, n)
    if b is None: abort(404)
    return Response(b, mimetype="image/png", headers={"Cache-Control": "max-age=3600"})
@app.post("/api/doc/<i>/ask")
def ask(i):
    need(i); q = (request.get_json(silent=True) or {}).get("q", "").strip()
    m = E.read_json(os.path.join(E.doc_dir(i), "meta.json"), {})
    if m.get("status") != "ready": return jsonify(error="This document is not ready yet."), 409
    if not q: return jsonify(error="Type a question."), 400
    try: r = E.ask(i, q)
    except SystemExit as e: return jsonify(error=str(e)), 429
    except Exception as e: return jsonify(error=f"Could not answer: {e}"), 500
    p = os.path.join(E.doc_dir(i), "chat.json"); c = E.read_json(p, []); c.append(r); E.write_json(p, c)
    return jsonify(r)

def main(port=7860):
    os.makedirs(E.LIB, exist_ok=True); E.resume_unfinished()
    print(f"Pagewise on http://localhost:{port}  (library: {E.LIB})")
    app.run(host="0.0.0.0", port=port, threaded=True)
