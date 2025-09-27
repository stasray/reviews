#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def load_scenario(path: Optional[str], base_url: Optional[str]) -> Dict[str, Any]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        data = {"base_url": base_url or "https://example.com", "steps": [{"name": "Home", "method": "GET", "path": "/", "weight": 1}]}
    if base_url:
        data["base_url"] = base_url
    # Normalize weights
    steps = data.get("steps", [])
    for s in steps:
        s.setdefault("weight", 1)
        s.setdefault("headers", {})
    return data


class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._next = time.perf_counter()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.perf_counter()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(self._next + self.interval, time.perf_counter())


def pick_step(steps: List[Dict[str, Any]]):
    weights = [max(1, int(s.get("weight", 1))) for s in steps]
    return random.choices(steps, weights=weights, k=1)[0]


def run_request(session: requests.Session, base_url: str, step: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    method = step.get("method", "GET").upper()
    url = base_url.rstrip("/") + step.get("path", "/")
    headers = step.get("headers", {})
    data = step.get("data", None)
    t0 = time.perf_counter()
    try:
        resp = session.request(method, url, headers=headers, data=data, timeout=timeout)
        dt = time.perf_counter() - t0
        size = int(resp.headers.get("content-length", 0)) or len(resp.content or b"")
        return {"ok": resp.ok, "status": resp.status_code, "latency": dt, "bytes": size, "name": step.get("name", url)}
    except requests.RequestException:
        dt = time.perf_counter() - t0
        return {"ok": False, "status": 0, "latency": dt, "bytes": 0, "name": step.get("name", url)}


def summarize(results: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
    latencies = [r["latency"] for r in results]
    ok_count = sum(1 for r in results if r["ok"]) 
    status_counts = Counter(r["status"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)
    per_name = defaultdict(list)
    for r in results:
        per_name[r["name"]].append(r["latency"])
    def pct(xs, p):
        if not xs:
            return 0.0
        return float(statistics.quantiles(xs, n=100, method="inclusive")[min(max(int(p),1),99)-1])
    summary = {
        "requests": len(results),
        "success": ok_count,
        "success_rate": round(ok_count / max(1, len(results)), 4),
        "rps": round(len(results) / max(1e-9, duration), 2),
        "latency_avg_ms": round(1000 * (sum(latencies) / max(1, len(latencies))), 2) if latencies else 0.0,
        "latency_p50_ms": round(1000 * pct(latencies, 50), 2) if latencies else 0.0,
        "latency_p90_ms": round(1000 * pct(latencies, 90), 2) if latencies else 0.0,
        "latency_p99_ms": round(1000 * pct(latencies, 99), 2) if latencies else 0.0,
        "status": dict(status_counts),
        "bytes_total": total_bytes,
        "bytes_per_sec": round(total_bytes / max(1e-9, duration), 2),
        "per_step": {
            name: {
                "count": len(v),
                "avg_ms": round(1000 * (sum(v) / max(1, len(v))), 2),
                "p90_ms": round(1000 * pct(v, 90), 2) if v else 0.0,
            }
            for name, v in per_name.items()
        }
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Simple web performance/load test")
    parser.add_argument("--scenario", default="perfomance/scenario_example.json", help="Path to scenario JSON")
    parser.add_argument("--base-url", default=None, help="Override base URL from scenario")
    parser.add_argument("--rate", type=float, default=10.0, help="Target request rate (RPS)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max in-flight requests")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration (seconds)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout (seconds)")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario if args.scenario and Path(args.scenario).exists() else None, args.base_url)
    base_url = scenario["base_url"]
    steps = scenario["steps"]

    limiter = RateLimiter(args.rate)
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    start = time.perf_counter()
    end = start + args.duration

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        with requests.Session() as session:
            futures: List[cf.Future] = []

            def submit_one():
                step = pick_step(steps)
                return pool.submit(run_request, session, base_url, step, args.timeout)

            # Prime some requests
            while time.perf_counter() < end and len(futures) < args.concurrency:
                limiter.wait()
                futures.append(submit_one())

            # Main loop: pace submissions and collect completed
            while time.perf_counter() < end:
                # Pacing
                limiter.wait()
                # Submit new
                futures.append(submit_one())
                # Drain any done futures quickly
                still: List[cf.Future] = []
                for fut in futures:
                    if fut.done():
                        try:
                            res = fut.result()
                            with results_lock:
                                results.append(res)
                        except Exception:
                            pass
                    else:
                        still.append(fut)
                futures = still

            # After duration, wait for queued to finish
            for fut in cf.as_completed(futures, timeout=args.timeout + 5):
                try:
                    res = fut.result()
                    with results_lock:
                        results.append(res)
                except Exception:
                    pass

    total_time = time.perf_counter() - start
    summary = summarize(results, total_time)
    print("Web performance summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
