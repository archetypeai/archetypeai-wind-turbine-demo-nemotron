"""Fault briefs from NVIDIA Nemotron (default) or Newton C 2.6.

When Omega + KNN commits a turbine to `fault`, `fault_brief()` summarises the
flagged window against the healthy peer turbine over the same hours and asks a
reasoning model for a short operator brief: what changed, the likely cause, what
to check. The model only sees SCADA statistics and the KNN vote — never the
status logs, so the cause it names is inferred, not looked up.

  model="nemotron"  NVIDIA Nemotron on NVIDIA's hosted API (OpenAI-compatible)
  model="newton"    Newton C 2.6 on Archetype's /query (system prompt in instruction_prompt)

Both get the same prompt and the same validation.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache

import pandas as pd

from data_loader import discover_turbines, load_turbine_window

DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEWTON_MODEL = "Newton::c2_6_8b_fp8_260424d7a55d5e"
MODELS = ("nemotron", "newton")
TIMEOUT = 60

# A turbine is "stopped in wind" when there is enough wind to generate but it doesn't.
CUT_IN_MPS = 4.0
IDLE_KW = 10.0

STATS_COLUMNS: dict[str, str] = {
    "wind_mps": "Wind speed (m/s)",
    "power_kw": "Power (kW)",
    "rotor_rpm": "Rotor speed (RPM)",
    "pitch_deg": "Blade angle (pitch position) A (°)",
    "gear_oil_c": "Gear oil temperature (°C)",
    "gen_brg_front_c": "Generator bearing front temperature (°C)",
    "grid_hz": "Grid frequency (Hz)",
}

SYSTEM_PROMPT = """You are a reliability engineer for an onshore wind farm (Senvion MM82 turbines, 2,050 kW rated).

An embedding classifier has flagged one turbine's ~21-hour SCADA window as FAULT. You get that turbine's window statistics and a healthy peer turbine on the same site over the same hours (same wind), plus the classifier's KNN vote.

Write a short brief for the operator. Return ONLY a JSON object, no prose or code fences:
{"observation": "...", "likely_cause": "...", "checks": ["...", "..."]}

Rules:
- observation: the clearest difference from the peer, citing numbers from the data (e.g. "WT01 0 kW vs WT09 1,640 kW at 11.2 m/s"). Max 140 characters.
- likely_cause: the most specific hypothesis the numbers support, phrased as a hypothesis ("consistent with...", "suggests..."). Don't just restate that it isn't generating. Max 120 characters.
- checks: 2-3 concrete things to inspect, each max 60 characters.
- Only use the signals given. Do not invent alarms, codes or readings.
- Round numbers for reading at a glance (kW and °C to whole numbers, m/s and rpm to one decimal).

Reading SCADA signals:
- Pitch near 90° = blades feathered (parked); near 0° = blades working. Rotor rpm near 0 with wind above cut-in = stopped, not idling.
- Gear-oil / bearing temperatures well below the peer mean the drivetrain has been cold for hours, i.e. a long stop rather than a brief one.
- Grid frequency matching the peer and near 50 Hz points away from a grid-side outage, toward a turbine-side trip (controller, converter, pitch, safety chain).
- Output low but rotor turning and pitch near the peer suggests curtailment or derating rather than a trip."""


def _endpoint() -> str:
    return (os.getenv("NVIDIA_API_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def model() -> str:
    return os.getenv("NEMOTRON_MODEL") or DEFAULT_MODEL


def enabled(which: str = "nemotron") -> bool:
    if which == "newton":
        return bool(os.getenv("ATAI_API_KEY"))
    return bool(os.getenv("NVIDIA_API_KEY"))


@lru_cache(maxsize=4)
def _frame(wt_id: str, start: str, end: str) -> pd.DataFrame:
    info = {t.wt_id: t for t in discover_turbines()}[wt_id]
    df = load_turbine_window(info, start, end, max_rows=20_000)
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    return df


def window_stats(wt_id: str, start: str, end: str, data_start: str, data_end: str) -> dict:
    """Summary statistics for one turbine over [start, end) at 10-minute cadence."""
    df = _frame(wt_id, data_start, data_end)
    w = df[(df["Date and time"] >= pd.to_datetime(start)) & (df["Date and time"] < pd.to_datetime(end))]
    out: dict = {"rows": len(w)}
    for key, col in STATS_COLUMNS.items():
        s = pd.to_numeric(w[col], errors="coerce").dropna()
        if s.empty:
            continue
        digits = 0 if key == "power_kw" else 2 if key == "grid_hz" else 1
        out[key] = {"mean": round(float(s.mean()), digits), "min": round(float(s.min()), digits),
                    "max": round(float(s.max()), digits)}
    wind = pd.to_numeric(w["Wind speed (m/s)"], errors="coerce")
    power = pd.to_numeric(w["Power (kW)"], errors="coerce")
    windy = wind >= CUT_IN_MPS
    if windy.any():
        stopped = windy & (power <= IDLE_KW)
        out["hours_with_wind_above_cut_in"] = round(float(windy.sum()) / 6, 1)
        out["hours_stopped_in_wind"] = round(float(stopped.sum()) / 6, 1)
    return out


def _chat(system: str, user: str, max_tokens: int = 600) -> str:
    body = json.dumps({
        "model": model(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # Turns off Nemotron 3's reasoning trace so the reply is just the JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{_endpoint()}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            data = json.load(res)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Nemotron {exc.code}: {exc.read().decode(errors='ignore')[:300]}") from exc
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def _newton_chat(system: str, user: str, max_new_tokens: int = 600) -> str:
    """Newton C 2.6 on /query via the same official client the Omega calls use."""
    from newton_client import _get_client

    client = _get_client()
    payload = client.requests_post(
        f"{client.api_endpoint}/query",
        data_payload=json.dumps({
            "query": user,
            "instruction_prompt": system,  # C 2.6 ignores the legacy system_prompt field
            "file_ids": [],
            "model": NEWTON_MODEL,
            "max_new_tokens": max_new_tokens,
        }),
        additional_headers={"Content-Type": "application/json"},
    )
    response = payload.get("response")
    if isinstance(response, dict):
        response = response.get("response")
    if isinstance(response, list):
        response = response[0] if response else ""
    return response if isinstance(response, str) else ""


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in Nemotron reply: {text[:200]!r}")
    obj = json.loads(text[start:end + 1])
    checks = [str(c) for c in obj.get("checks") or [] if str(c).strip()][:3]
    if not obj.get("observation") or not obj.get("likely_cause") or not checks:
        raise ValueError(f"Incomplete Nemotron brief: {obj!r}")
    return {"observation": str(obj["observation"]), "likely_cause": str(obj["likely_cause"]), "checks": checks}


def fault_brief(turbine: str, peer: str, window_start: str, window_end: str, votes: dict,
                data_start: str, data_end: str, which: str = "nemotron") -> dict:
    """Ask the chosen model for an operator brief on a flagged window. Returns the parsed brief."""
    payload = {
        "flagged_turbine": f"WT{turbine}",
        "healthy_peer": f"WT{peer}",
        "window": {"start": window_start, "end": window_end},
        "knn_vote": votes,
        f"WT{turbine}": window_stats(turbine, window_start, window_end, data_start, data_end),
        f"WT{peer}": window_stats(peer, window_start, window_end, data_start, data_end),
    }
    # Time the model call alone (not the SCADA stats above, not parsing) so the two models compare fairly.
    started = time.perf_counter()
    if which == "newton":
        reply, model_id = _newton_chat(SYSTEM_PROMPT, json.dumps(payload)), NEWTON_MODEL
    else:
        reply, model_id = _chat(SYSTEM_PROMPT, json.dumps(payload)), model()
    latency_ms = round((time.perf_counter() - started) * 1000)
    return {**_parse(reply), "model": model_id, "which": which, "latency_ms": latency_ms}
