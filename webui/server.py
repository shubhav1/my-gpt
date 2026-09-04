"""Flask app tying the training runner, checkpoints, and chat inference together behind a small
HTTP/SSE API, plus serving the static frontend. Launch via ../run_webui.py.
"""
import json
import os

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from . import checkpoints, config, inference
from .training_runner import run as training_run

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(config.STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(config.STATIC_DIR, filename)


@app.route("/api/config")
def api_config():
    return jsonify(config.DEFAULT_TRAIN_CONFIG)


@app.route("/api/train/status")
def api_train_status():
    return jsonify(
        {
            "status": training_run.status,
            "run_name": training_run.run_name,
            "error": training_run.error_message,
            "history": training_run.history,
        }
    )


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    overrides = request.get_json(force=True, silent=True) or {}
    try:
        training_run.start(overrides)
    except (RuntimeError, ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"status": "started"})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    training_run.stop()
    return jsonify({"status": "stopping"})


@app.route("/api/train/stream")
def api_train_stream():
    q = training_run.subscribe()

    def event_stream():
        try:
            while True:
                event = q.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "stopped", "error"):
                    break
        finally:
            training_run.unsubscribe(q)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/checkpoints")
def api_checkpoints():
    return jsonify(checkpoints.list_checkpoints())


@app.route("/api/models")
def api_models():
    return jsonify(checkpoints.list_models())


@app.route("/api/checkpoints/detail")
def api_checkpoint_detail():
    filename = request.args.get("filename")
    location = request.args.get("location")
    if not filename or location not in ("saved", "temp"):
        return jsonify({"error": "filename and location are required"}), 400

    try:
        detail = checkpoints.get_detail(filename, location)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(detail)


@app.route("/api/checkpoints/save", methods=["POST"])
def api_checkpoints_save():
    body = request.get_json(force=True, silent=True) or {}
    filename = body.get("filename")
    if not filename:
        return jsonify({"error": "no checkpoint filename given"}), 400

    try:
        saved_filename = checkpoints.promote_to_saved(filename)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify({"filename": saved_filename, "location": "saved"})


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    body = request.get_json(force=True, silent=True) or {}
    run_name = body.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name is required"}), 400

    try:
        removed = checkpoints.delete_model(run_name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify({"deleted": removed})


@app.route("/api/models/rename", methods=["POST"])
def api_models_rename():
    body = request.get_json(force=True, silent=True) or {}
    old_name = body.get("run_name")
    new_name = body.get("new_run_name")
    if not old_name or not new_name:
        return jsonify({"error": "run_name and new_run_name are required"}), 400

    try:
        renamed_to = checkpoints.rename_model(old_name, new_name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409

    return jsonify({"run_name": renamed_to})


@app.route("/api/session/cleanup", methods=["POST"])
def api_session_cleanup():
    # Fired via navigator.sendBeacon when a tab closes: anything not explicitly saved is
    # discarded. Also runs on server startup (see main()) to catch crash-orphaned temp files.
    checkpoints.clear_temp_checkpoints()
    return jsonify({"status": "cleared"})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True, silent=True) or {}
    filename = body.get("checkpoint")
    location = body.get("location", "saved")
    prompt = body.get("prompt", "")
    max_new_tokens = max(1, min(int(body.get("max_new_tokens", 200)), 500))

    if not filename:
        return jsonify({"error": "no checkpoint selected"}), 400

    def event_stream():
        try:
            for chunk in inference.generate_stream(filename, location, prompt, max_new_tokens):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


def main():
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.TEMP_CHECKPOINTS_DIR, exist_ok=True)
    checkpoints.clear_temp_checkpoints()  # fresh session boundary; sweeps up any crash leftovers
    print("my-gpt web UI: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, threaded=True)


if __name__ == "__main__":
    main()
