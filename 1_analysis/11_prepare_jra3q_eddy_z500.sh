#!/usr/bin/env bash
set -euo pipefail

# 本脚本只在用户指定的数据目录中创建新文件；已有完整输出默认直接复用。
: "${CHW_DATA_ROOT:?Set CHW_DATA_ROOT to the analysis-ready data directory.}"

raw_dir="${JRA3Q_RAW_DIR:-${CHW_DATA_ROOT}/JRA3Q/eddy_z500_jja_1981_2014_raw}"
grid_file="${JRA3Q_GRID_FILE:-${CHW_DATA_ROOT}/static/grid1x1.txt}"
output_file="${JRA3Q_OUTPUT_FILE:-${CHW_DATA_ROOT}/JRA3Q/JRA3Q_hgt500_JJA_monthly_1x1_1981-2014.nc}"
jobs="${JRA3Q_DOWNLOAD_JOBS:-8}"

if [[ -s "$output_file" && "${JRA3Q_FORCE:-0}" != "1" ]]; then
    printf 'SKIP existing output %s\n' "$output_file"
    exit 0
fi
if [[ ! -s "$grid_file" ]]; then
    printf 'ERROR: missing CDO grid file: %s\n' "$grid_file" >&2
    exit 1
fi

mkdir -p "$raw_dir" "$(dirname "$output_file")"
url_list="$(mktemp "${raw_dir}/jra3q_urls.XXXXXX")"
tmp_file="$(mktemp "$(dirname "$output_file")/jra3q_output.XXXXXX.nc")"
cleanup() {
    rm -f "$url_list" "$tmp_file"
}
trap cleanup EXIT

for year in $(seq 1981 2014); do
    for month in 06 07 08; do
        case "$month" in
            06) end_day=30 ;;
            07|08) end_day=31 ;;
        esac
        start_time="${year}${month}0100"
        end_time="${year}${month}${end_day}18"
        name="jra3q-ms-mn.anl_p.0_3_5.hgt-pres-an-gauss-mn.${start_time}_${end_time}.nc"
        printf '%s\n' "https://osdf-data.gdex.ucar.edu/ncar/gdex/d640002/anl_p/${year}${month}/${name}" >> "$url_list"
    done
done

download_one() {
    url="$1"
    destination="$2"
    name="${url##*/}"
    output="${destination}/${name}"
    partial="${output}.part"
    if [[ -s "$output" ]]; then
        printf 'SKIP %s\n' "$name"
        return 0
    fi
    wget --no-check-certificate -c -O "$partial" "$url"
    mv "$partial" "$output"
    printf 'DONE %s\n' "$name"
}
export -f download_one
xargs -P "$jobs" -n 1 -I '{}' bash -c 'download_one "$1" "$2"' _ '{}' "$raw_dir" < "$url_list"

mapfile -t files < <(find "$raw_dir" -maxdepth 1 -type f -name '*.nc' -size +0c | sort)
if [[ "${#files[@]}" -ne 102 ]]; then
    printf 'ERROR: expected 102 complete JRA-3Q JJA files, found %s\n' "${#files[@]}" >&2
    exit 1
fi

cdo -O -f nc4 -z zip_4 remapbil,"$grid_file" -sellevel,500 -cat "${files[@]}" "$tmp_file"

python3 - "$tmp_file" <<'PY'
import sys
import xarray as xr

path = sys.argv[1]
with xr.open_dataset(path) as dataset:
    key = "hgt-pres-an-gauss-mn"
    if key not in dataset:
        raise SystemExit(f"missing variable: {key}")
    data = dataset[key]
    if data.sizes.get("time") != 102:
        raise SystemExit(f"expected 102 months, got {data.sizes.get('time')}")
    years = data.time.dt.year.values
    months = data.time.dt.month.values
    if int(years.min()) != 1981 or int(years.max()) != 2014:
        raise SystemExit("unexpected year range")
    if set(int(value) for value in months) != {6, 7, 8}:
        raise SystemExit("unexpected month set")
PY

mv "$tmp_file" "$output_file"
trap - EXIT
rm -f "$url_list"
printf 'READY %s\n' "$output_file"
