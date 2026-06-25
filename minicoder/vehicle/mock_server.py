"""FastAPI 教学型 Mock Vehicle API；不应直接用于真实车辆控制。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from ..intent import get_timezone
from .errors import VehicleError
from .models import (
    normalize_location_name,
    normalize_vehicle_id,
    validate_coordinates,
    validate_radius,
    validate_temperature,
)
from .store import FaultSpec, VehicleStateStore

LOGGER = logging.getLogger("minicoder.vehicle_api")
SCHEMA_VERSION = 1


class RouteRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=120)
    origin: str | None = Field(default=None, min_length=1, max_length=100)
    vehicle_id: str | None = None
    departure_time: str | None = Field(default=None, max_length=80)

    @field_validator("destination")
    @classmethod
    def destination_valid(cls, value: str) -> str:
        return normalize_location_name(value, field="destination", max_length=120)

    @field_validator("origin")
    @classmethod
    def origin_valid(cls, value: str | None) -> str | None:
        return normalize_location_name(value, field="origin") if value is not None else None

    @field_validator("vehicle_id")
    @classmethod
    def vehicle_valid(cls, value: str | None) -> str | None:
        return normalize_vehicle_id(value) if value is not None else None

    @model_validator(mode="after")
    def origin_available(self) -> RouteRequest:
        if self.origin is None and self.vehicle_id is None:
            raise ValueError("origin 与 vehicle_id 至少提供一个")
        return self


class ClimateRequest(BaseModel):
    action: Literal["turn_on", "turn_off", "set_temperature"]
    target_temperature: float | None = None

    @model_validator(mode="after")
    def temperature_valid(self) -> ClimateRequest:
        if self.action == "set_temperature":
            self.target_temperature = validate_temperature(self.target_temperature)
        elif self.target_temperature is not None:
            raise ValueError("仅 set_temperature 可以提供 target_temperature")
        return self


class FaultRequest(BaseModel):
    mode: Literal["rate_limit", "timeout", "server_error", "stale_data", "vehicle_offline"]
    remaining: int = Field(default=1, ge=1, le=100)
    delay_seconds: float = Field(default=0.05, ge=0, le=5)
    retry_after: float = Field(default=0.01, ge=0, le=60)


def create_app(
    *,
    store: VehicleStateStore | None = None,
    token: str = "local-demo-token",
    test_mode: bool = False,
    timezone_name: str = "Asia/Shanghai",
) -> FastAPI:
    state = store or VehicleStateStore()
    zone = get_timezone(timezone_name)
    app = FastAPI(
        title="minicoder Mock Vehicle API",
        version="1.0.0",
        description=(
            "仅用于教学、测试和本地演示。它不提供真实车辆认证、持久化、分布式一致性或生产安全保障。"
        ),
    )
    app.state.vehicle_store = state
    app.state.test_mode = test_mode

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID", "")[:64] or uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                ensure_ascii=False,
            )
        )
        return response

    @app.exception_handler(VehicleError)
    async def vehicle_error(request: Request, error: VehicleError) -> JSONResponse:
        return _error_response(
            request,
            status_code=error.status_code or 400,
            code=error.code,
            message=str(error),
            retryable=error.retryable,
            retry_after=error.retry_after,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        message = error.errors()[0].get("msg", "请求参数无效") if error.errors() else "请求参数无效"
        return _error_response(
            request,
            status_code=422,
            code="VALIDATION_ERROR",
            message=str(message),
        )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if authorization != f"Bearer {token}":
            from .errors import VehicleAuthenticationError

            raise VehicleAuthenticationError(
                "车辆 API 认证失败",
                code="AUTHENTICATION_FAILED",
                status_code=401,
            )

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION, "service": "mock-vehicle-api"}

    @app.get("/v1/vehicles/{vehicle_id}/status", dependencies=[Depends(authorize)])
    async def vehicle_status(vehicle_id: str, request: Request) -> dict[str, Any]:
        fault = await _apply_fault(state)
        vehicle = state.get_vehicle(normalize_vehicle_id(vehicle_id))
        observed = datetime.now(zone)
        stale = bool(fault and fault.mode == "stale_data")
        if stale:
            observed -= timedelta(hours=2)
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.state.request_id,
            "observed_at": observed.isoformat(),
            "stale": stale,
            "vehicle": vehicle,
        }

    @app.post("/v1/routes/estimate", dependencies=[Depends(authorize)])
    async def estimate_route(body: RouteRequest, request: Request) -> dict[str, Any]:
        await _apply_fault(state)
        route = state.estimate_route(
            destination=body.destination,
            origin=body.origin,
            vehicle_id=body.vehicle_id,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.state.request_id,
            "route": route,
        }

    @app.get("/v1/charging-stations", dependencies=[Depends(authorize)])
    async def charging_stations(
        request: Request,
        location: str | None = Query(default=None, min_length=1, max_length=100),
        corridor: list[str] | None = Query(default=None),
        vehicle_id: str | None = Query(default=None),
        latitude: float | None = Query(default=None),
        longitude: float | None = Query(default=None),
        radius_km: float = Query(default=20.0),
    ) -> dict[str, Any]:
        await _apply_fault(state)
        if location is not None:
            location = normalize_location_name(location)
        if vehicle_id is not None:
            vehicle_id = normalize_vehicle_id(vehicle_id)
        if latitude is not None or longitude is not None:
            validate_coordinates(latitude, longitude)
            validate_radius(radius_km)
        stations = state.find_charging_stations(
            location=location,
            corridor=corridor,
            vehicle_id=vehicle_id,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.state.request_id,
            "stations": stations,
        }

    @app.put("/v1/vehicles/{vehicle_id}/climate", dependencies=[Depends(authorize)])
    async def control_climate(
        vehicle_id: str,
        body: ClimateRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        await _apply_fault(state)
        result = state.update_climate(
            normalize_vehicle_id(vehicle_id),
            action=body.action,
            target_temperature=body.target_temperature,
            idempotency_key=idempotency_key or "",
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.state.request_id,
            "result": result,
        }

    if test_mode:

        @app.post("/__test__/faults", dependencies=[Depends(authorize)])
        async def configure_fault(body: FaultRequest) -> dict[str, Any]:
            state.set_fault(FaultSpec(**body.model_dump()))
            return {"status": "configured", "mode": body.mode, "remaining": body.remaining}

        @app.post("/__test__/reset", dependencies=[Depends(authorize)])
        async def reset() -> dict[str, str]:
            state.reset()
            return {"status": "reset"}

    return app


async def _apply_fault(store: VehicleStateStore) -> FaultSpec | None:
    fault = store.consume_fault()
    if fault is None or fault.mode == "stale_data":
        return fault
    if fault.mode == "timeout":
        await asyncio.sleep(fault.delay_seconds)
        return fault
    if fault.mode == "vehicle_offline":
        from .errors import VehicleConflictError

        raise VehicleConflictError(
            "模拟车辆离线",
            code="VEHICLE_OFFLINE",
            status_code=409,
        )
    if fault.mode == "rate_limit":
        from .errors import VehicleRateLimitError

        raise VehicleRateLimitError(
            "模拟车辆 API 限流",
            code="RATE_LIMITED",
            status_code=429,
            retryable=True,
            retry_after=fault.retry_after,
        )
    from .errors import VehicleServiceError

    raise VehicleServiceError(
        "模拟车辆 API 服务异常",
        code="SERVICE_UNAVAILABLE",
        status_code=503,
        retryable=True,
    )


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    retry_after: float | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    headers = {"X-Request-ID": request_id}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动教学型 Mock Vehicle API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("教学型 Mock API 只允许绑定 localhost")

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_mode = os.environ.get("MINICODER_MOCK_TEST_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    token = os.environ.get("MINICODER_VEHICLE_API_TOKEN", "").strip() or "local-demo-token"
    uvicorn.run(
        create_app(token=token, test_mode=test_mode),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
