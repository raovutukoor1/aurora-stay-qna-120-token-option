#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETWORK_NAME="${DOCKER_NETWORK_NAME:-aurora-stays-network}"
BACKEND_NAME="${BACKEND_CONTAINER_NAME:-aurora-stays-backend}"
FRONTEND_NAME="${FRONTEND_CONTAINER_NAME:-aurora-stays-frontend}"
BACKEND_IMAGE="${BACKEND_IMAGE_NAME:-aurora-stays-backend}"
FRONTEND_IMAGE="${FRONTEND_IMAGE_NAME:-aurora-stays-frontend}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"

usage() {
    printf 'Usage: %s {start|stop|test}\n' "${0##*/}"
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        printf 'Error: Docker is not installed or not on PATH.\n' >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        printf 'Error: the Docker daemon is not available.\n' >&2
        exit 1
    fi
}

start() {
    require_docker

    if docker inspect "$BACKEND_NAME" >/dev/null 2>&1; then
        while IFS= read -r entry; do
            case "$entry" in
                OPENAI_API_KEY=*)
                    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
                        OPENAI_API_KEY="${entry#*=}"
                    fi
                    ;;
                OPENAI_BASE_URL=*)
                    if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
                        OPENAI_BASE_URL="${entry#*=}"
                    fi
                    ;;
            esac
        done < <(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$BACKEND_NAME")
    fi

    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        printf 'Error: OPENAI_API_KEY is not set. Configure it as a Codespaces secret and retry.\n' >&2
        exit 1
    fi
    export OPENAI_API_KEY
    if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
        export OPENAI_BASE_URL
    fi

    docker build --tag "$BACKEND_IMAGE" "$ROOT_DIR/backend"
    docker build --tag "$FRONTEND_IMAGE" "$ROOT_DIR/frontend"

    if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
        docker network create "$NETWORK_NAME" >/dev/null
    fi

    docker rm --force "$FRONTEND_NAME" "$BACKEND_NAME" >/dev/null 2>&1 || true
    docker run --detach \
        --name "$BACKEND_NAME" \
        --network host \
        --env OPENAI_API_KEY \
        --env OPENAI_BASE_URL \
        "$BACKEND_IMAGE"

    if ! curl --fail --silent --show-error \
        --retry 45 --retry-delay 2 --retry-connrefused --retry-all-errors --retry-max-time 100 \
        "http://127.0.0.1:7860/" >/dev/null; then
        printf 'Error: backend did not become ready. Recent backend logs:\n' >&2
        docker logs --tail 60 "$BACKEND_NAME" >&2 || true
        exit 1
    fi

    docker run --detach \
        --name "$FRONTEND_NAME" \
        --network "$NETWORK_NAME" \
        --add-host host.docker.internal:host-gateway \
        --publish "${FRONTEND_PORT}:8501" \
        --env "BACKEND_URL=http://host.docker.internal:7860" \
        "$FRONTEND_IMAGE"

    if ! curl --fail --silent --show-error \
        --retry 30 --retry-delay 2 --retry-connrefused --retry-all-errors --retry-max-time 65 \
        "http://127.0.0.1:${FRONTEND_PORT}/_stcore/health" >/dev/null; then
        printf 'Error: frontend did not become ready. Recent frontend logs:\n' >&2
        docker logs --tail 60 "$FRONTEND_NAME" >&2 || true
        exit 1
    fi

    printf 'Aurora Stays is running.\nAPI:      http://localhost:7860\nFrontend: http://localhost:%s\n' \
        "$FRONTEND_PORT"
}

stop() {
    require_docker
    docker rm --force "$FRONTEND_NAME" "$BACKEND_NAME" >/dev/null 2>&1 || true
    printf 'Aurora Stays containers stopped.\n'
}

test_api() {
    require_docker
    if ! docker inspect "$FRONTEND_NAME" >/dev/null 2>&1; then
        printf 'Error: frontend is not running. Start it with %s start.\n' "${0##*/}" >&2
        exit 1
    fi

    if ! docker exec "$FRONTEND_NAME" python -c '
import requests
import sys

base_url = "http://host.docker.internal:7860"
health = requests.get(f"{base_url}/", timeout=10)
health.raise_for_status()
response = requests.post(
    f"{base_url}/v1/answer",
    json={
        "query": "What is the cancellation policy for a flexible reservation?",
        "k": 5,
        "temperature": 0.0,
        "max_tokens": 128,
    },
    timeout=120,
)
response.raise_for_status()
answer = response.json().get("answer", "")
if not isinstance(answer, str) or not answer.strip():
    print("Error: API response did not contain an answer.", file=sys.stderr)
    sys.exit(1)
if answer.startswith("Local smoke-test fallback:"):
    print(f"Error: backend returned its local fallback: {answer}", file=sys.stderr)
    sys.exit(1)
print(answer)
'; then
        printf 'Error: sample query from the frontend container failed.\n' >&2
        exit 1
    fi
    printf 'Frontend health check and end-to-end sample query passed.\n'
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    test)
        test_api
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac