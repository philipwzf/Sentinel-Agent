#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time


def _ensure_x11_socket_dir() -> None:
    path = "/tmp/.X11-unix"
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return
    try:
        os.chmod(path, 0o1777)
    except OSError:
        pass


def _terminate(proc: subprocess.Popen | None) -> None:
    if not proc or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    _ensure_x11_socket_dir()

    display = os.environ.get("DISPLAY", ":99")
    xvfb_cmd = [
        "Xvfb",
        display,
        "-screen",
        "0",
        "1024x768x24",
        "-ac",
        "+extension",
        "GLX",
        "+render",
        "-noreset",
    ]
    xvfb_proc = subprocess.Popen(xvfb_cmd)
    os.environ["DISPLAY"] = display

    time.sleep(0.5)
    if xvfb_proc.poll() is not None:
        return xvfb_proc.returncode or 1

    server_cmd = ["uv", "run", "src/server.py", *sys.argv[1:]]
    server_proc = subprocess.Popen(server_cmd)

    def _handle_exit(_signum, _frame):
        _terminate(server_proc)
        _terminate(xvfb_proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_exit)
    signal.signal(signal.SIGINT, _handle_exit)

    exit_code = server_proc.wait()
    _terminate(xvfb_proc)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
