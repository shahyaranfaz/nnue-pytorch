#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/d/nnue/robotmoon}
JOBS=${JOBS:-3}

mkdir -p \
    "$ROOT/hard_relabel" \
    "$ROOT/farseer_relabel" \
    "$ROOT/leela96_relabel" \
    "$ROOT/.staging"

MANIFEST=$(mktemp)
trap 'rm -f "$MANIFEST"' EXIT

cat > "$MANIFEST" <<'EOF'
vondele/master-binpacks_relabel|5dda733e4253e3081e523f05b8d0509e71283945|dfrc_n5000.relabel-BT4-tf13tune.binpack|hard_relabel|40993832254
vondele/master-binpacks_relabel|5dda733e4253e3081e523f05b8d0509e71283945|multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack|hard_relabel|30487704890
vondele/master-binpacks_relabel|5dda733e4253e3081e523f05b8d0509e71283945|nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack|hard_relabel|44150410791
vondele/master-binpacks_relabel|5dda733e4253e3081e523f05b8d0509e71283945|wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack|hard_relabel|7800748146
vondele/from_kaggle_2_relabel|ff3efba57a093eac08486ed10fd22c37cdfc92da|T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack|farseer_relabel|22120303232
vondele/from_kaggle_2_relabel|ff3efba57a093eac08486ed10fd22c37cdfc92da|T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack|farseer_relabel|22145581828
vondele/from_kaggle_2_relabel|ff3efba57a093eac08486ed10fd22c37cdfc92da|T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack|farseer_relabel|22131380240
vondele/from_kaggle_2_relabel|ff3efba57a093eac08486ed10fd22c37cdfc92da|T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack|farseer_relabel|22118569308
vondele/from_kaggle_2_relabel|ff3efba57a093eac08486ed10fd22c37cdfc92da|T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack|farseer_relabel|22164105768
vondele/from_kaggle_1_relabel|7ddab4f3c9e9b48e8c6741c21ed8b7c044c7db06|leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack|leela96_relabel|19550050458
vondele/from_kaggle_1_relabel|7ddab4f3c9e9b48e8c6741c21ed8b7c044c7db06|leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack|leela96_relabel|19548579082
vondele/from_kaggle_1_relabel|7ddab4f3c9e9b48e8c6741c21ed8b7c044c7db06|leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack|leela96_relabel|19549769139
vondele/from_kaggle_1_relabel|7ddab4f3c9e9b48e8c6741c21ed8b7c044c7db06|leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack|leela96_relabel|19549070571
vondele/from_kaggle_1_relabel|7ddab4f3c9e9b48e8c6741c21ed8b7c044c7db06|leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack|leela96_relabel|19548894060
EOF

download_one() {
    IFS='|' read -r repo revision filename directory expected_size <<< "$1"

    destination="$ROOT/$directory/$filename"
    staging="$ROOT/.staging/$directory/$filename"
    staged_file="$staging/output/$filename"

    if [[ -f "$destination" ]]; then
        actual_size=$(stat -c '%s' "$destination")
        if [[ "$actual_size" == "$expected_size" ]]; then
            echo "Already complete: $filename"
            return 0
        fi

        echo "Wrong-size completed file: $destination" >&2
        echo "Expected $expected_size, got $actual_size" >&2
        return 1
    fi

    mkdir -p "$staging/output"
    echo "Downloading: $filename"

    completed=0
    for attempt in 1 2 3; do
        if hf download "$repo" "$filename" \
               --repo-type dataset \
               --revision "$revision" \
               --local-dir "$staging/output"
        then
            completed=1
            break
        fi

        echo "Attempt $attempt failed: $filename" >&2
        sleep 5
    done

    if [[ "$completed" != 1 ]]; then
        echo "DOWNLOAD FAILURE: $filename" >&2
        return 1
    fi

    if [[ ! -f "$staged_file" ]]; then
        echo "MISSING STAGED FILE: $staged_file" >&2
        return 1
    fi

    actual_size=$(stat -c '%s' "$staged_file")
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "SIZE FAILURE: $filename" >&2
        echo "Expected $expected_size, got $actual_size" >&2
        return 1
    fi

    mkdir -p "$ROOT/$directory"
    mv "$staged_file" "$destination"

    final_size=$(stat -c '%s' "$destination")
    if [[ "$final_size" != "$expected_size" ]]; then
        echo "FINAL SIZE FAILURE: $filename" >&2
        return 1
    fi

    echo "Verified: $filename"
}

export ROOT
export -f download_one

xargs -P "$JOBS" -d '\n' -I '{}' bash -c 'download_one "$1"' _ '{}' < "$MANIFEST"

echo
echo "Final verification:"
failed=0

while IFS='|' read -r _repo _revision filename directory expected_size; do
    path="$ROOT/$directory/$filename"

    if [[ ! -f "$path" ]]; then
        echo "MISSING  $path"
        failed=1
        continue
    fi

    actual_size=$(stat -c '%s' "$path")
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "BAD SIZE $path expected=$expected_size actual=$actual_size"
        failed=1
    else
        echo "OK       $path"
    fi
done < "$MANIFEST"

echo
df -h /mnt/d

if [[ "$failed" != 0 ]]; then
    echo "One or more files failed verification." >&2
    exit 1
fi

echo "All 14 files match their historical sizes."
