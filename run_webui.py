"""Launches the training/chat UI for my-gpt.

Run from this directory (so my_gpt.py and BPE/ resolve on sys.path):

    python run_webui.py

Then open http://127.0.0.1:5050. See webui/README.md for details.
"""
from webui.server import main

if __name__ == "__main__":
    main()
