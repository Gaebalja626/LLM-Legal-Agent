#!/usr/bin/env python3
"""
PostToolUse hook for legal-ir session logging.

Reads tool-use JSON from stdin and appends a timestamped line to
~/legal-ir/SESSION_LOG.md describing the call.
Never raises — designed to be safe for Claude Code hook pipeline.
"""
import sys, json, time, os, re

LOG_PATH = os.path.expanduser("~/legal-ir/SESSION_LOG.md")
MAX_BODY = 400


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    tool = data.get("tool_name", "?")
    ti = data.get("tool_input") or {}

    body = ""
    extra = ""
    if tool == "Bash":
        body = (ti.get("command", "") or "")[:MAX_BODY]
        body = re.sub(r"\s+", " ", body).strip()
        if ti.get("run_in_background"):
            extra = " [bg]"
    elif tool in ("Edit", "Write", "NotebookEdit"):
        body = ti.get("file_path", "") or ti.get("notebook_path", "") or ""
        if tool == "Edit":
            old = (ti.get("old_string", "") or "")[:60]
            old = re.sub(r"\s+", " ", old).strip()
            if old:
                extra = f"  | edited: {old}…"
        elif tool == "Write":
            extra = f"  | wrote {len(ti.get('content','') or '')} chars"
    else:
        return  # ignore other tools

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"- {ts}  [{tool}]{extra}  `{body}`\n"
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        return


if __name__ == "__main__":
    main()
