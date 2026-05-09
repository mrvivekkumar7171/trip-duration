# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.5
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files and buffering stdout/stderr to avoid situations where the
# application crashes without emitting any logs due to buffering.
ENV PYTHONDONTWRITEBYTECODE=1\
    PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Create a non-privileged user that the app will run under. https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Switch to the non-privileged user to run the application.
USER appuser

# Copy required files to run the application inside app directory in the container and ensure they are owned by the non-root user.
COPY --chown=appuser:appuser app.py requirements.txt /models/model.joblib src/ /app/

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 8080

# Run app.py when the container launches
# CMD ["python", "app.py"]
# CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8000"]
CMD ["gunicorn", "app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]