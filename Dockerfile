FROM python:3 as builder

RUN mkdir /install
WORKDIR /install

# Copy ca.cer (certificate authority) if it exists. Necessary in a SSL decrypt evironment.
COPY requirements.txt ca.cer* /

RUN apt-get update -y && \
    apt-get install -y git && \
    (test ! -f /ca.cer || git config --global http.sslCAInfo /ca.cer) && \
    (test ! -f /ca.cer || pip config set global.cert /ca.cer) && \
    pip install --prefix=/install -r /requirements.txt

FROM python:3-slim

ARG version_number
ARG commit_sha

ENV VERSION_NUMBER=$version_number
ENV COMMIT_SHA=$commit_sha

COPY --from=builder /install /usr/local
COPY bottomdetection /app/bottomdetection

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR /app

CMD ["python", "/app/bottomdetection/docker_main.py"]
