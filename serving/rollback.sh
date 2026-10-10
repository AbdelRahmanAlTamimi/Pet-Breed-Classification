#!/usr/bin/env bash
set -euo pipefail

compose=(docker compose -f docker/docker-compose.canary.yml)
sed -i -E 's/server model-stable:8000 weight=[0-9]+;/server model-stable:8000 weight=100;/' serving/nginx/nginx.conf
sed -i -E 's/server model-canary:8000 weight=[0-9]+;/server model-canary:8000 weight=5 down;/' serving/nginx/nginx.conf
"${compose[@]}" exec nginx nginx -t
"${compose[@]}" exec nginx nginx -s reload
