#!/usr/bin/env python3
"""Drive the gem5 guest terminal like gem5term: wait for the shell prompt,
send each command, wait for the prompt again. Logs everything received."""

import re
import socket
import sys
import time

port = int(sys.argv[1])
log = open(sys.argv[2], "ab", buffering=0)
PROMPT = re.compile(rb"bash-[0-9.]+# ")
YN = re.compile(rb"\[Y/N\] ?$")
# Optional: argv[3] = "resume" -> discard replayed history, answer Y first.
RESUME = len(sys.argv) > 3 and sys.argv[3] == "resume"
COMMANDS = [
    ("rocm-smi --showcomputepartition", 3600),
    ("rocm-smi --showmemorypartition", 3600),
    ("rocm-smi --setmemorypartition NPS4", 7200),
    ("rocm-smi --showmemorypartition", 3600),
    ("rocm-smi --setcomputepartition CPX", 3600),
    ("rocm-smi --showcomputepartition", 3600),
    ("exit", 0),
]
if RESUME:
    COMMANDS = [("Y", 7200)] + COMMANDS[3:]

for _ in range(600):
    try:
        s = socket.create_connection(("localhost", port))
        break
    except OSError:
        time.sleep(2)
else:
    sys.exit("could not connect")
s.settimeout(5)
buf = b""


def wait_prompt(timeout):
    global buf
    end = time.time() + timeout
    while time.time() < end:
        if YN.search(buf.rstrip(b"\r\n") + b" "):
            stamp("answering Y")
            s.sendall(b"Y\n")
            buf = b""
            continue
        m = PROMPT.search(buf)
        if m:
            buf = buf[m.end() :]
            return True
        try:
            d = s.recv(65536)
        except TimeoutError:
            continue
        if not d:
            return False
        log.write(d)
        buf += d
    return False


def stamp(msg):
    log.write(f"\n##DRIVER {time.strftime('%H:%M:%S')} {msg}\n".encode())


if RESUME:
    # Drain the history gem5 replays on connect.
    end = time.time() + 15
    while time.time() < end:
        try:
            d = s.recv(65536)
        except TimeoutError:
            break
        log.write(d)
    stamp("resume: drained history")
elif not wait_prompt(4 * 3600):
    stamp("never saw initial prompt")
    sys.exit(1)
for cmd, timeout in COMMANDS:
    time.sleep(1)
    stamp(f"send: {cmd}")
    s.sendall(cmd.encode() + b"\n")
    if timeout and not wait_prompt(timeout):
        stamp(f"timeout/closed waiting after: {cmd}")
        sys.exit(2)
stamp("done")
try:
    while True:
        d = s.recv(65536)
        if not d:
            break
        log.write(d)
except OSError:
    pass
