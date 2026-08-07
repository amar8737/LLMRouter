"""Health and metrics endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import Depends, Request

from llmrouterx.router.llmrouter import LLMRouter


def _aggregate_per_provider(
    snapshot: dict[str, Any],
    providers: list[Any],
    provider_health: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Aggregate labeled metrics into per-provider summaries.

    Input: snapshot from MetricsCollector (counters, labeled_counters, timings, labeled_timings)
    Output: {
        "providers": {
            "openai": {
                "request_count": 10234,
                "success_count": 10156,
                "error_count": 78,
                "error_rate": 0.0076,
                "success_rate": 0.9924,
                "latency_stats": {...percentiles...},
                "error_breakdown": {"rate_limit_error": 35, ...}
            },
            ...
        },
        "global": {
            "healthy_providers": 4,
            "total_providers": 4,
            "success_rate": 0.987,
            "latency_stats": {...percentiles...},
            "total_errors": 662
        }
    }
    """

    result: dict[str, Any] = {"providers": {}, "global": {}}

    labeled_counters = snapshot.get("labeled_counters", {})
    labeled_timings = snapshot.get("labeled_timings", {})
    counters = snapshot.get("counters", {})
    timings = snapshot.get("timings", {})

    # -----------------------------------------------------------------------
    # Per-Provider Aggregation
    # -----------------------------------------------------------------------

    provider_names = [getattr(p, "name", None) for p in providers]
    provider_names = [n for n in provider_names if n]

    for provider_name in provider_names:
        label_key = f"provider={provider_name}"

        # Request counts
        request_count = labeled_counters.get("request.count", {}).get(label_key, 0)
        success_count = labeled_counters.get("request.success", {}).get(label_key, 0)
        error_count = labeled_counters.get("request.error", {}).get(label_key, 0)

        # Latency percentiles
        latency_values = labeled_timings.get("request.latency", {}).get(label_key, [])
        latency_stats = _compute_percentiles(latency_values)

        # Error breakdown (per provider)
        error_breakdown = {}
        for label, count in labeled_counters.get("request.error", {}).items():
            if label.startswith(f"{label_key},"):
                # Parse "provider=openai,error_type=rate_limit_error"
                parts = dict(p.split("=") for p in label.split(","))
                error_type = parts.get("error_type", "unknown")
                error_breakdown[error_type] = count

        error_rate = error_count / request_count if request_count else 0.0
        success_rate = success_count / request_count if request_count else 0.0

        result["providers"][provider_name] = {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "latency_stats": latency_stats,
            "error_breakdown": error_breakdown,
        }

    # -----------------------------------------------------------------------
    # Global Aggregation
    # -----------------------------------------------------------------------

    total_requests = counters.get("request.count", 0)
    total_success = counters.get("request.success", 0)
    total_errors = counters.get("request.error", 0)

    all_latency_values = timings.get("request.latency", [])
    global_latency_stats = _compute_percentiles(all_latency_values)

    global_error_rate = total_errors / total_requests if total_requests else 0.0
    global_success_rate = total_success / total_requests if total_requests else 0.0

    healthy_providers = sum(1 for v in (provider_health or {}).values() if v)
    total_providers = len(provider_health or {})

    # Global error breakdown
    global_error_breakdown = {}
    for label, count in counters.get("request.error", {}).items():
        if label.startswith("error_type="):
            error_type = label.split("=")[1]
            global_error_breakdown[error_type] = count

    result["global"] = {
        "healthy_providers": healthy_providers,
        "total_providers": total_providers,
        "request_count": total_requests,
        "success_count": total_success,
        "error_count": total_errors,
        "error_rate": global_error_rate,
        "success_rate": global_success_rate,
        "latency_stats": global_latency_stats,
        "error_breakdown": global_error_breakdown,
    }

    return result


def _compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute latency percentiles from a list of float values (seconds)."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    from statistics import mean

    return {
        "count": n,
        "min": round(sorted_vals[0] * 1000, 2),
        "max": round(sorted_vals[-1] * 1000, 2),
        "mean": round(mean(sorted_vals) * 1000, 2),
        "median": round(sorted_vals[n // 2] * 1000, 2),
        "p50": round(sorted_vals[int(n * 0.50)] * 1000, 2),
        "p95": round(sorted_vals[int(n * 0.95)] * 1000, 2),
        "p99": round(sorted_vals[int(n * 0.99)] * 1000, 2),
    }


def _collect_circuit_breaker_state(providers: list[Any]) -> dict[str, Any]:
    """Collect circuit breaker state from all provider routers and their clients."""
    result: dict[str, list[dict[str, Any]]] = {}
    for provider in providers:
        provider_name = getattr(provider, "name", None)
        if not provider_name:
            continue

        clients = getattr(provider, "clients", [])
        if not clients:
            continue

        result[provider_name] = []
        for client in clients:
            cb = getattr(client, "circuit_breaker", None)
            if cb:
                result[provider_name].append(
                    {
                        "api_key_suffix": getattr(client, "api_key", "")[-4:]
                        if getattr(client, "api_key", "")
                        else "unknown",
                        "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
                        "failure_count": cb.failure_count,
                        "half_open_calls": getattr(cb, "_half_open_calls", 0),
                        "half_open_successes": getattr(cb, "_half_open_successes", 0),
                    }
                )
    return result


def create_health_endpoint(start_time: float, health_timeout: float | None):
    """Create health check endpoint."""

    async def health_check(request: Request) -> dict[str, Any]:
        rt: LLMRouter = request.app.state.llm_router

        import asyncio

        if health_timeout is not None:
            try:
                provider_health = await asyncio.wait_for(rt.health(), timeout=health_timeout)
            except asyncio.TimeoutError:
                provider_health = {}
        else:
            provider_health = await rt.health()

        total = len(provider_health)
        healthy = sum(1 for value in provider_health.values() if value)
        return {
            "status": "ok" if total and healthy == total else "degraded",
            "providers": provider_health,
            "healthy_count": healthy,
            "total_providers": total,
            "uptime_seconds": time.monotonic() - start_time,
        }

    return health_check


def create_metrics_endpoint(start_time: float, admin_guard: Any):
    """Create metrics endpoint."""

    async def metrics(request: Request) -> dict[str, Any]:
        rt: LLMRouter = request.app.state.llm_router
        snapshot = rt.get_metrics()

        # Compute derived metrics for dashboard
        provider_health = await rt.health()
        derived = _aggregate_per_provider(snapshot, rt.providers, provider_health)

        # Collect circuit breaker state
        circuit_breakers = _collect_circuit_breaker_state(rt.providers)

        return {
            "snapshot": snapshot,
            "derived": derived,
            "circuit_breakers": circuit_breakers,
            "uptime_seconds": time.monotonic() - start_time,
            "timestamp": time.time(),
        }

    # Apply admin guard
    metrics.__annotations__ = {"request": Request, "return": dict[str, Any]}
    return Depends(admin_guard)(metrics)
