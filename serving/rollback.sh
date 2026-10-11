#!/usr/bin/env bash
set -euo pipefail

# nginx.conf is bind-mounted as a single file, so it must be rewritten in place:
# `sed -i` replaces the inode and the container would keep serving the old config.
conf=serving/nginx/nginx.conf
compose=(docker compose -f docker/docker-compose.canary.yml)
updated=$(sed -E \
    -e 's/server model-stable:8000 weight=[0-9]+;/server model-stable:8000 weight=100;/' \
    -e 's/server model-canary:8000 weight=[0-9]+;/server model-canary:8000 weight=5 down;/' \
    "$conf")
printf '%s\n' "$updated" > "$conf"
"${compose[@]}" exec nginx nginx -t
"${compose[@]}" exec nginx nginx -s reload
