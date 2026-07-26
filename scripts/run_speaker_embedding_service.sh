#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
data_dir="${PODTRANS_DATA_DIR:-${project_root}/data}"
python_bin="${PODTRANS_PYTHON:-${project_root}/.venv/bin/python}"
speaker_socket="${SPEAKER_SERVICE_SOCKET:-/tmp/podtrans-speaker-embedding.sock}"
device="${SPEAKER_DEVICE:-cuda:0}"
batch_size="${SPEAKER_BATCH_SIZE:-512}"
warmup_batch_size="${SPEAKER_WARMUP_BATCH_SIZE:-${batch_size}}"
cpu_threads="${SPEAKER_SERVICE_CPU_THREADS:-8}"
handler_workers="${SPEAKER_HANDLER_WORKERS:-4}"
max_queued_requests="${SPEAKER_MAX_QUEUED_REQUESTS:-8}"
batch_wait_ms="${SPEAKER_BATCH_WAIT_MS:-50}"

export PODTRANS_DATA_DIR="${data_dir}"
mkdir -p "${data_dir}/_metadata"

log_file="${data_dir}/_metadata/speaker_embedding_service.log"
status_file="${data_dir}/_metadata/speaker_embedding_service_status.json"
printf '[service] %s starting speaker embedding service batch=%s device=%s\n' \
  "$(date -Is)" "${batch_size}" "${device}" > "${log_file}"

exec "${python_bin}" "${project_root}/speaker_embedding_service.py" \
  --device "${device}" \
  --socket "${speaker_socket}" \
  --status-file "${status_file}" \
  --max-batch-size "${batch_size}" \
  --warmup-batch-size "${warmup_batch_size}" \
  --cpu-threads "${cpu_threads}" \
  --handler-workers "${handler_workers}" \
  --max-queued-requests "${max_queued_requests}" \
  --batch-wait-ms "${batch_wait_ms}" \
  >> "${log_file}" 2>&1
