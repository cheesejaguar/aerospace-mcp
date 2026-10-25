"""Smoke-test an installed aerospace-mcp (a wheel, not the source checkout).

Run it with the Python of the environment the package was installed into,
from a directory outside the repository so the checkout cannot shadow the
installed package::

    uv run --no-project --with dist/aerospace_mcp-*.whl python scripts/smoke_installed.py

It checks what a user's one-line install relies on: the native extension
loads from site-packages, and the three console scripts (CLI, MCP stdio
server, HTTP server) start and answer real requests. Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.request
from pathlib import Path

BIN = Path(sysconfig.get_path("scripts"))


def _script(name: str) -> str:
    path = shutil.which(name, path=str(BIN))
    assert path, f"console script {name!r} not installed in {BIN}"
    return path


def check_package() -> None:
    import aerospace_mcp
    from aerospace_mcp import _native

    location = Path(aerospace_mcp.__file__).resolve()
    assert "site-packages" in location.parts, f"not the installed package: {location}"
    print(f"package: {location.parent}")
    print(f"native:  {Path(_native.__file__).name}")


def check_cli() -> None:
    cli = _script("aerospace-mcp-cli")
    # (argv, text a successful run prints): tools report failures as text,
    # so the exit code alone does not prove the tool worked.
    runs = [
        # Native orbit core
        (
            ["run", "hohmann_transfer", "--r1_m", "6778000", "--r2_m", "42164000"],
            '"total_delta_v_ms"',
        ),
        # Native Lambert solver
        (
            [
                "run",
                "lambert_problem_solver",
                "--r1_m",
                "[7000000,0,0]",
                "--r2_m",
                "[0,7000000,0]",
                "--tof_s",
                "3600",
            ],
            "Transfer angle: 90.000",
        ),
        # Native rocket simulation
        (
            [
                "run",
                "rocket_3dof_trajectory",
                "--rocket_geometry",
                '{"length_m":2,"diameter_m":0.15,"dry_mass_kg":10,'
                '"propellant_mass_kg":5,"thrust_curve":[[0,500],[10,500],[10.1,0]]}',
                "--launch_conditions",
                '{"launch_angle_deg":85}',
            ],
            "Max Altitude",
        ),
        # OpenAP + airport data
        (
            [
                "run",
                "plan_flight",
                "--departure",
                '{"city":"San Jose","iata":"SJC"}',
                "--arrival",
                '{"city":"Tokyo","iata":"NRT"}',
                "--aircraft",
                '{"ac_type":"A359"}',
            ],
            '"block"',
        ),
    ]
    for argv, expected in runs:
        proc = subprocess.run(
            [cli, *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert expected in out and "Traceback" not in out, out
        print(f"cli {argv[1]}: ok")


def _rpc(proc: subprocess.Popen, message: dict) -> dict | None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    if "id" not in message:
        return None
    while True:
        line = proc.stdout.readline()
        assert line, "MCP server closed stdout"
        reply = json.loads(line)
        if reply.get("id") == message["id"]:
            assert "error" not in reply, reply
            return reply["result"]


def check_mcp_stdio() -> None:
    proc = subprocess.Popen(
        [_script("aerospace-mcp")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        init = _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke", "version": "1"},
                },
            },
        )
        assert init["serverInfo"]["name"] == "aerospace-mcp", init
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in tools["tools"]}
        assert {"hohmann_transfer", "plan_flight", "search_aerospace_tools"} <= names
        result = _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "hohmann_transfer",
                    "arguments": {"r1_m": 6778000, "r2_m": 42164000},
                },
            },
        )
        assert not result.get("isError"), result
        text = result["content"][0]["text"]
        assert abs(json.loads(text)["total_delta_v_ms"] - 3854.0) < 1.0, text
        print(f"mcp stdio: ok ({len(names)} tools)")
    finally:
        proc.kill()
        proc.wait()


def check_http() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "AEROSPACE_MCP_HOST": "127.0.0.1",
        "AEROSPACE_MCP_PORT": str(port),
    }
    proc = subprocess.Popen(
        [_script("aerospace-mcp-http")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, proc.stderr.read()
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
                    health = json.load(resp)
                break
            except OSError:
                assert time.monotonic() < deadline, "HTTP server did not start"
                time.sleep(0.5)
        assert health["status"] == "ok", health
        request = urllib.request.Request(
            f"{base}/plan",
            data=json.dumps(
                {"depart_city": "San Jose", "arrive_city": "Tokyo", "ac_type": "A359"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as resp:
            plan = json.load(resp)
        assert plan["distance_nm"] > 4000, plan
        print(
            f"http: ok ({plan['depart']['iata']}->{plan['arrive']['iata']} {plan['distance_nm']:.0f} nm)"
        )
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    check_package()
    check_cli()
    check_mcp_stdio()
    check_http()
    print("installed package smoke test passed")
    sys.exit(0)
