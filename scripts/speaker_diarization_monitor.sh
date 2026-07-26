#!/usr/bin/env bash
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
data_dir="${PODTRANS_DATA_DIR:-${project_root}/data}"
python_bin="${PODTRANS_PYTHON:-${project_root}/.venv/bin/python}"

export PODTRANS_DATA_DIR="${data_dir}"
worker_pid_file="${1:?usage: $0 WORKER_PID_FILE [INTERVAL_SECONDS]}"
interval="${2:-1800}"
pid_file="${data_dir}/_metadata/speaker_diarization_monitor.pid"
status_file="${data_dir}/_metadata/speaker_diarization_monitor.status"
report_file="${data_dir}/_metadata/speaker_diarization_report.json"
manifest_file="${data_dir}/_metadata/speaker_diarization_manifest.jsonl"
service_status_file="${data_dir}/_metadata/speaker_embedding_service_status.json"

printf '%s\n' "$$" > "${pid_file}"
printf 'running worker_pid_file=%s started_at=%s\n' \
  "${worker_pid_file}" "$(date -Is)" > "${status_file}"

any_worker_alive() {
  local shard pid
  while read -r shard pid; do
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done < "${worker_pid_file}"
  return 1
}

snapshot() {
  local shard pid
  echo
  echo "[monitor] $(date -Is)"
  while read -r shard pid; do
    ps -p "${pid}" -o pid=,etime=,%cpu=,%mem=,stat=,cmd= 2>/dev/null \
      | sed "s/^/[monitor] worker=${shard} /" || true
  done < "${worker_pid_file}"

  if [ -s "${report_file}" ]; then
    jq -r \
      '"[monitor] report eligible=\(.eligible) ok=\(.speaker_ok) error=\(.speaker_error) pending=\(.pending)"' \
      "${report_file}" 2>/dev/null || true
  fi

  if [ -s "${service_status_file}" ]; then
    jq -r \
      '"[monitor] embedding state=\(.state) pid=\(.pid) precision=\(.speaker_model_precision) reserved_mb=\(.max_peak_reserved_mb)"' \
      "${service_status_file}" 2>/dev/null || true
  fi

  if [ -s "${manifest_file}" ]; then
    tail -1 "${manifest_file}" \
      | jq -r '"[monitor] last_manifest=\(.time) \(.status) \(.path)"' 2>/dev/null \
      || true
  fi

  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null \
    | sed 's/^/[monitor] gpu=/' \
    || true

  for worker_log in "${data_dir}"/_metadata/speaker_diarization_worker_*.log; do
    [ -s "${worker_log}" ] || continue
    tail -1 "${worker_log}" | sed "s|^|[monitor] ${worker_log}=|" || true
  done
}

snapshot
while any_worker_alive; do
  elapsed=0
  while [ "${elapsed}" -lt "${interval}" ] && any_worker_alive; do
    sleep 30
    elapsed=$((elapsed + 30))
  done
  if any_worker_alive; then
    "${python_bin}" "${project_root}/diarize_funasr.py" \
      --verify-only >/dev/null 2>&1 || true
    snapshot
  fi
done

echo
echo "[monitor] $(date -Is) all workers exited; running final verification"
if "${python_bin}" "${project_root}/diarize_funasr.py" --verify-only; then
  verify_exit=0
else
  verify_exit=$?
fi
echo "[monitor] final_verify_exit=${verify_exit}"
snapshot
printf 'stopped verify_exit=%s stopped_at=%s\n' \
  "${verify_exit}" "$(date -Is)" > "${status_file}"
