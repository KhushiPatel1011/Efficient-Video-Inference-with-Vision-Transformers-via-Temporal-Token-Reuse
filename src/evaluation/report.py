from typing import Dict, List


def summarize_run(rows: List[Dict]) -> Dict[str, float]:
    """
    This function helps in summarizing a run given per-frame rows containing 'latency_ms'.
    It returns average latency and FPS.
    """
    if len(rows) == 0:
        return {"num_frames": 0, "avg_latency_ms": 0.0, "fps": 0.0}

    latencies = [float(r["latency_ms"]) for r in rows if "latency_ms" in r]
    avg_latency = sum(latencies) / max(1, len(latencies))

    # FPS = frames / total_seconds
    total_sec = sum(latencies) / 1000.0
    fps = (len(latencies) / total_sec) if total_sec > 0 else 0.0

    return {
        "num_frames": float(len(latencies)),
        "avg_latency_ms": float(avg_latency),
        "fps": float(fps),
    }
