import sys
import time
import threading
import subprocess
import json
import glob
import ollama
from router.config import OLLAMA_MODEL, OLLAMA_URL


def _try_nvidia() -> float | None:
    """NVIDIA via nvidia-smi : works on Linux and Windows."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _try_amd() -> float | None:
    """
    AMD power, tried in order:
      1. sysfs hwmon       (Linux)
      2. amd-smi           (cross-platform)
      3. rocm-smi          (Linux fallback)
    """
    # 1. sysfs hwmon : Linux only
    if sys.platform != "win32":
        for power_file in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average"):
            try:
                with open(power_file) as f:
                    return float(f.read().strip()) / 1_000_000  # µW → W
            except (OSError, ValueError):
                continue

    # 2. amd-smi : AMD's cross-platform tool (Windows and Linux)
    try:
        result = subprocess.run(
            ["amd-smi", "metric", "--json"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list) and data:
                power = data[0].get("power", {})
                for key in ("average_socket_power", "socket_power", "current_socket_power"):
                    if key in power:
                        return float(str(power[key]).split()[0])
    except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired, KeyError):
        pass

    # 3. rocm-smi : Linux only
    if sys.platform != "win32":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for card_data in data.values():
                    for key, val in card_data.items():
                        if "power" in key.lower():
                            return float(str(val).split()[0])
        except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass

    return None


def _try_rapl() -> float | None:
    """Intel RAPL (Linux only): derive package power from two energy samples 100 ms apart."""
    if sys.platform == "win32":
        return None
    energy_file = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        with open(energy_file) as f:
            e1 = int(f.read())
        time.sleep(0.1)
        with open(energy_file) as f:
            e2 = int(f.read())
        return (e2 - e1) / 1e6 / 0.1  # µJ → W
    except (OSError, ValueError):
        pass
    return None


def _sample_power_watts() -> float | None:
    """
    Read current power draw in watts.
    Tries NVIDIA (nvidia-smi) -> AMD (sysfs / amd-smi / rocm-smi) -> Intel RAPL (Linux).
    Returns None if no supported hardware or tooling is detected.
    """
    for sampler in (_try_nvidia, _try_amd, _try_rapl):
        result = sampler()
        if result is not None:
            return result
    return None


def ask(prompt: str) -> tuple[str, dict]:
    """
    Send a prompt to the local Ollama model.
    Returns (response, energy_meta) where energy_meta contains:
        measured_wh  - marginal energy above idle (avg_w - idle_w * duration), or None
        idle_w       - power draw sampled before inference, or None
        avg_w        - mean power draw across samples taken during inference, or None
        peak_w       - max power draw observed during inference, or None
        duration_s   - inference duration in seconds
        sample_count - number of in-flight power samples collected
    """
    client = ollama.Client(host=OLLAMA_URL)

    watts_idle = _sample_power_watts()

    # Poll power on a background thread throughout inference
    samples: list[float] = []
    stop_event = threading.Event()

    def _poll() -> None:
        while not stop_event.is_set():
            w = _sample_power_watts()
            if w is not None:
                samples.append(w)
            stop_event.wait(0.5)  # sample every 500 ms

    poller = threading.Thread(target=_poll, daemon=True)

    t_start = time.monotonic()
    poller.start()
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    t_end = time.monotonic()

    stop_event.set()
    poller.join(timeout=2)

    duration_seconds = t_end - t_start

    if watts_idle is not None and samples:
        watts_avg = sum(samples) / len(samples)
        watts_peak = max(samples)
        wh_marginal = max(watts_avg - watts_idle, 0) * duration_seconds / 3600
    else:
        watts_avg = None
        watts_peak = None
        wh_marginal = None

    energy_meta = {
        "measured_wh": wh_marginal,
        "idle_w": watts_idle,
        "avg_w": watts_avg,
        "peak_w": watts_peak,
        "duration_s": duration_seconds,
        "sample_count": len(samples),
    }

    return response["response"], energy_meta
