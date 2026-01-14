FROM ghcr.io/astral-sh/uv:python3.11-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcursor1 \
        x11vnc \
        xauth \
        xvfb \
        xserver-xorg-core \
        xserver-xorg-video-dummy \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md startx.py ./
COPY src src
COPY SENTINEL_code SENTINEL_code
COPY examples examples

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["python3", "startx.py", "--host", "0.0.0.0"]
EXPOSE 9009
