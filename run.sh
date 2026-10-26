#!/bin/sh
cd "$(dirname "$0")"
# Settings (ANTHROPIC_API_KEY, BARKER_SOURCE, SMTP_*, ...) can live in .env next to this script.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec "${UV:-uv}" run --with-requirements requirements.txt -m barker "$@"
