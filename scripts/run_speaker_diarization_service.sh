#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
data_dir="${PODTRANS_DATA_DIR:-${project_root}/data}"
python_bin="${PODTRANS_PYTHON:-${project_root}/.venv/bin/python}"
worker_count="${SPEAKER_WORKERS:-4}"
worker_start_stagger="${SPEAKER_WORKER_START_STAGGER:-2}"
monitor_interval="${SPEAKER_MONITOR_INTERVAL:-1800}"
speaker_socket="${SPEAKER_SERVICE_SOCKET:-/tmp/podtrans-speaker-embedding.sock}"
embedding_unit="${SPEAKER_EMBEDDING_UNIT:-podtrans-speaker-embedding.service}"
speaker_batch_size="${SPEAKER_BATCH_SIZE:-512}"
job_batch_size="${SPEAKER_JOB_BATCH_SIZE:-4}"
decode_workers="${SPEAKER_DECODE_WORKERS:-4}"
cpu_threads="${SPEAKER_WORKER_CPU_THREADS:-4}"
device="${SPEAKER_DEVICE:-cuda:0}"
vad_device="${SPEAKER_VAD_DEVICE:-cpu}"

export PODTRANS_DATA_DIR="${data_dir}"
mkdir -p "${data_dir}/_metadata"

worker_pid_file="${data_dir}/_metadata/speaker_diarization_worker_pids"
monitor_pid_file="${data_dir}/_metadata/speaker_diarization_monitor.pid"
service_status_file="${data_dir}/_metadata/speaker_diarization_service.status"
main_log="${data_dir}/_metadata/speaker_diarization_full.log"
monitor_log="${data_dir}/_metadata/speaker_diarization_monitor.log"

: > "${worker_pid_file}"
printf 'running started_at=%s supervisor_pid=%s workers=%s\n' \
  "$(date -Is)" "$$" "${worker_count}" > "${service_status_file}"
printf '\n[service] %s starting diarization workers=%s batch=%s\n' \
  "$(date -Is)" "${worker_count}" "${speaker_batch_size}" >> "${main_log}"

worker_pids=()
stop_workers() {
  local pid
  for pid in "${worker_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap stop_workers INT TERM

for ((shard_index = 0; shard_index < worker_count; shard_index++)); do
  worker_log="${data_dir}/_metadata/speaker_diarization_worker_${shard_index}.log"
  : > "${worker_log}"
  printf '\n[worker] %s shard=%s/%s starting\n' \
    "$(date -Is)" "${shard_index}" "${worker_count}" >> "${worker_log}"
  "${python_bin}" "${project_root}/diarize_funasr.py" \
    --device "${device}" \
    --vad-device "${vad_device}" \
    --speaker-service-socket "${speaker_socket}" \
    --speaker-service-timeout "${SPEAKER_SERVICE_TIMEOUT:-900}" \
    --sort "${SPEAKER_JOB_SORT:-size}" \
    --speaker-batch-size "${speaker_batch_size}" \
    --job-batch-size "${job_batch_size}" \
    --decode-workers "${decode_workers}" \
    --cpu-threads "${cpu_threads}" \
    --audio-chunk-seconds "${SPEAKER_AUDIO_CHUNK_SECONDS:-300}" \
    --merge-threshold "${SPEAKER_MERGE_THRESHOLD:-0.60}" \
    --report-every-batches "${SPEAKER_REPORT_EVERY_BATCHES:-25}" \
    --retry-errors \
    --num-shards "${worker_count}" \
    --shard-index "${shard_index}" \
    --no-report \
    >> "${worker_log}" 2>&1 &
  worker_pid=$!
  worker_pids+=("${worker_pid}")
  printf '%s %s\n' "${shard_index}" "${worker_pid}" >> "${worker_pid_file}"
  if [ "${shard_index}" -lt $((worker_count - 1)) ]; then
    sleep "${worker_start_stagger}"
  fi
done

printf '[service] %s workers_started pids=%s\n' \
  "$(date -Is)" "${worker_pids[*]}" >> "${main_log}"
printf '\n[service] %s monitoring workers interval=%ss\n' \
  "$(date -Is)" "${monitor_interval}" >> "${monitor_log}"
bash "${script_dir}/speaker_diarization_monitor.sh" \
  "${worker_pid_file}" "${monitor_interval}" >> "${monitor_log}" 2>&1 &
monitor_pid=$!
printf '%s\n' "${monitor_pid}" > "${monitor_pid_file}"

main_exit=0
for worker_pid in "${worker_pids[@]}"; do
  wait "${worker_pid}"
  worker_exit=$?
  if [ "${worker_exit}" -ne 0 ]; then
    main_exit="${worker_exit}"
    printf '[service] %s worker_pid=%s exit=%s\n' \
      "$(date -Is)" "${worker_pid}" "${worker_exit}" >> "${main_log}"
  fi
done

wait "${monitor_pid}"
monitor_exit=$?
printf 'stopped stopped_at=%s main_exit=%s monitor_exit=%s\n' \
  "$(date -Is)" "${main_exit}" "${monitor_exit}" > "${service_status_file}"

if [ "${STOP_EMBEDDING_SERVICE:-1}" = "1" ]; then
  systemctl --user stop "${embedding_unit}" 2>/dev/null || true
fi
exit "${main_exit}"
