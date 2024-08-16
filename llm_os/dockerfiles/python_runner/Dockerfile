# This is the only file from the AgentRun repository in this section
# This file was taken from https://github.com/Jonathan-Adly/AgentRun/blob/81ec7acdec27c3673b08a5a21a84b22aabbe188e/agentrun-api/docker/code_runner/Dockerfile
# The above two lines and this line has been added into this Dockerfile for acknowledgement purposes
FROM python:3.12.2-slim-bullseye

# Security best practices
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Create a non-root user with its home directory
RUN useradd --create-home pythonuser
USER pythonuser
WORKDIR /code

# Optional: Only if you have specific dependencies
# COPY --chown=appuser:appuser requirements.txt .
# RUN pip install --user -r requirements.txt
