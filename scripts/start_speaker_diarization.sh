#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
data_dir="${PODTRANS_DATA_DIR:-${project_root}/data}"
python_bin="${PODTRANS_PYTHON:-${project_root}/.venv/bin/python}"
pipeline_unit="${SPEAKER_PIPELINE_UNIT:-podtrans-speaker-diarization.service}"
embedding_unit="${SPEAKER_EMBEDDING_UNIT:-podtrans-speaker-embedding.service}"
speaker_socket="${SPEAKER_SERVICE_SOCKET:-/tmp/podtrans-speaker-embedding.sock}"
worker_count="${SPEAKER_WORKERS:-4}"
speaker_batch_size="${SPEAKER_BATCH_SIZE:-512}"
warmup_batch_size="${SPEAKER_WARMUP_BATCH_SIZE:-${speaker_batch_size}}"
job_batch_size="${SPEAKER_JOB_BATCH_SIZE:-4}"
decode_workers="${SPEAKER_DECODE_WORKERS:-4}"
worker_cpu_threads="${SPEAKER_WORKER_CPU_THREADS:-4}"
service_cpu_threads="${SPEAKER_SERVICE_CPU_THREADS:-8}"
handler_workers="${SPEAKER_HANDLER_WORKERS:-4}"
max_queued_requests="${SPEAKER_MAX_QUEUED_REQUESTS:-8}"
device="${SPEAKER_DEVICE:-cuda:0}"
vad_device="${SPEAKER_VAD_DEVICE:-cpu}"

export PODTRANS_DATA_DIR="${data_dir}"

if systemctl --user is-active --quiet "${pipeline_unit}"; then
  main_pid="$(systemctl --user show "${pipeline_unit}" --property=MainPID --value)"
  echo "[launcher] already running unit=${pipeline_unit} supervisor_pid=${main_pid}"
  exit 0
fi

systemctl --user stop "${embedding_unit}" 2>/dev/null || true
for unit in "${pipeline_unit}" "${embedding_unit}"; do
  if systemctl --user is-failed --quiet "${unit}"; then
    systemctl --user reset-failed "${unit}"
  fi
done
rm -f "${speaker_socket}"

systemd-run --user \
  --unit="${embedding_unit%.service}" \
  --collect \
  --property=Type=exec \
  --property=WorkingDirectory="${project_root}" \
  --property=KillMode=control-group \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PODTRANS_DATA_DIR="${data_dir}" \
  --setenv=PODTRANS_PYTHON="${python_bin}" \
  --setenv=SPEAKER_SERVICE_SOCKET="${speaker_socket}" \
  --setenv=SPEAKER_BATCH_SIZE="${speaker_batch_size}" \
  --setenv=SPEAKER_WARMUP_BATCH_SIZE="${warmup_batch_size}" \
  --setenv=SPEAKER_SERVICE_CPU_THREADS="${service_cpu_threads}" \
  --setenv=SPEAKER_HANDLER_WORKERS="${handler_workers}" \
  --setenv=SPEAKER_MAX_QUEUED_REQUESTS="${max_queued_requests}" \
  --setenv=SPEAKER_DEVICE="${device}" \
  /usr/bin/bash "${script_dir}/run_speaker_embedding_service.sh"

embedding_ready=0
for _ in $(seq 1 180); do
  if "${python_bin}" "${project_root}/speaker_embedding_service.py" \
    --check \
    --socket "${speaker_socket}" \
    --timeout-seconds 5 >/dev/null 2>&1; then
    embedding_ready=1
    break
  fi
  if ! systemctl --user is-active --quiet "${embedding_unit}"; then
    break
  fi
  sleep 1
done
if [ "${embedding_ready}" -ne 1 ]; then
  echo "[launcher] ${embedding_unit} did not become ready"
  journalctl --user-unit="${embedding_unit}" --no-pager -n 40
  exit 1
fi

systemd-run --user \
  --unit="${pipeline_unit%.service}" \
  --collect \
  --property=Type=exec \
  --property=WorkingDirectory="${project_root}" \
  --property=KillMode=control-group \
  --property=Requires="${embedding_unit}" \
  --property=After="${embedding_unit}" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PODTRANS_DATA_DIR="${data_dir}" \
  --setenv=PODTRANS_PYTHON="${python_bin}" \
  --setenv=SPEAKER_SERVICE_SOCKET="${speaker_socket}" \
  --setenv=SPEAKER_EMBEDDING_UNIT="${embedding_unit}" \
  --setenv=SPEAKER_WORKERS="${worker_count}" \
  --setenv=SPEAKER_BATCH_SIZE="${speaker_batch_size}" \
  --setenv=SPEAKER_JOB_BATCH_SIZE="${job_batch_size}" \
  --setenv=SPEAKER_DECODE_WORKERS="${decode_workers}" \
  --setenv=SPEAKER_WORKER_CPU_THREADS="${worker_cpu_threads}" \
  --setenv=SPEAKER_DEVICE="${device}" \
  --setenv=SPEAKER_VAD_DEVICE="${vad_device}" \
  /usr/bin/bash "${script_dir}/run_speaker_diarization_service.sh"

sleep 2
if ! systemctl --user is-active --quiet "${pipeline_unit}"; then
  echo "[launcher] ${pipeline_unit} did not remain active"
  journalctl --user-unit="${pipeline_unit}" --no-pager -n 40
  exit 1
fi

supervisor_pid="$(systemctl --user show "${pipeline_unit}" --property=MainPID --value)"
embedding_pid="$(systemctl --user show "${embedding_unit}" --property=MainPID --value)"
echo "[launcher] embedding_unit=${embedding_unit} model_pid=${embedding_pid}"
echo "[launcher] pipeline_unit=${pipeline_unit} supervisor_pid=${supervisor_pid}"
echo "[launcher] workers=${worker_count} speaker_batch_size=${speaker_batch_size}"
echo "[launcher] data_dir=${data_dir}"
echo "[launcher] main_log=${data_dir}/_metadata/speaker_diarization_full.log"
echo "[launcher] monitor_log=${data_dir}/_metadata/speaker_diarization_monitor.log"
