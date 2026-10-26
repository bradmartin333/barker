FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY barker ./barker
# Bake the YAMNet model into the image (checksummed), so a fresh container needs no download.
RUN python -m barker.yamnet

ENV PYTHONUNBUFFERED=1
# The heartbeat file is touched for every second of audio received, so a dead or missing mic stream turns the
# container unhealthy, which the homelab healthcheck reports.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os, sys, time; sys.exit(time.time() - os.path.getmtime('/tmp/barker-heartbeat') > 60)"
CMD ["python", "-m", "barker"]
