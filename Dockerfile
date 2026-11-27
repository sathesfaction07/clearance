# The handbook is fetched while building, so each image carries one fixed version of the
# corpus: what was searched can always be traced to the image that searched it.
FROM python:3.12-slim AS corpus
WORKDIR /build
COPY scripts/fetch_handbook.py scripts/
RUN python scripts/fetch_handbook.py

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLEARANCE_CORPUS=/srv/handbook \
    CLEARANCE_ACCESS=/etc/clearance/access.yaml
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin clearance
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf /root/.cache
COPY config/access.yaml /etc/clearance/access.yaml
COPY data/questions.yaml /app/questions.yaml
COPY --from=corpus /build/data/raw/handbook/md /srv/handbook
USER 10001
EXPOSE 8080
# --no-access-log: uvicorn's access log would record the query string, and questions about
# leave, pay or health are personal. The service logs its own line, without the query.
CMD ["uvicorn", "--factory", "clearance.service:from_env", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
