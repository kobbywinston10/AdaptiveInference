#!/usr/bin/env bash

set -u

BASE="/var/scratch/$USER/adaptiveInference"
FULL="$BASE/ChestXray14/full"
TMP="$BASE/download_tmp"
RESIZER="$HOME/adaptiveInference/scripts/downsample_nih_archive.py"

mkdir -p "$FULL" "$TMP"

URLS=(
"https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
"https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz"
"https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz"
"https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz"
"https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz"
"https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz"
"https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz"
"https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz"
"https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz"
"https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz"
"https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz"
"https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"
)

# Archive 001 has already been processed, so begin at index 2.
for N in $(seq 2 12); do

    NUM=$(printf "%03d" "$N")
    IDX=$((N - 1))
    URL="${URLS[$IDX]}"

    ARCHIVE="$TMP/images_${NUM}.tar.gz"
    OUTPUT="$FULL/images_${NUM}"

    echo
    echo "=================================================="
    echo "Processing images_${NUM}"
    echo "=================================================="

    mkdir -p "$OUTPUT"

    # Download only if the archive is not already present.
    echo "Downloading/resuming images_${NUM}..."
    wget -c "$URL" -O "$ARCHIVE"

    echo "Checking archive..."
    if ! gzip -t "$ARCHIVE"; then
        echo "ERROR: images_${NUM}.tar.gz failed gzip validation."
        exit 1
    fi

    SOURCE_COUNT=$(tar -tzf "$ARCHIVE" | grep -Ei '\.png$' | wc -l)

    echo "PNG files in source archive: $SOURCE_COUNT"

    ATTEMPT=1

    while true; do

        OUTPUT_COUNT=$(find "$OUTPUT" -type f -name '*.png' | wc -l)

        echo "Current resized images: $OUTPUT_COUNT / $SOURCE_COUNT"

        if [ "$OUTPUT_COUNT" -eq "$SOURCE_COUNT" ]; then
            echo "Archive images_${NUM} is complete."
            break
        fi

        echo "Resize attempt $ATTEMPT..."

        # srun may return non-zero if the 15-minute job reaches its time limit.
        # That is okay: rerunning is safe because the Python script skips
        # files that already exist.
        srun -N 1 -n 1 --time=00:15:00 \
        bash -lc "
            source /var/scratch/$USER/miniforge3/etc/profile.d/conda.sh
            conda activate /var/scratch/$USER/conda-envs/adaptive-inference

            export TMPDIR=/tmp/$USER-\$SLURM_JOB_ID
            mkdir -p \"\$TMPDIR\"

            cd $HOME/adaptiveInference

            python \"$RESIZER\" \
                \"$ARCHIVE\" \
                \"$OUTPUT\" \
                --size 224
        " || true

        ATTEMPT=$((ATTEMPT + 1))
    done

    FINAL_COUNT=$(find "$OUTPUT" -type f -name '*.png' | wc -l)

    if [ "$FINAL_COUNT" -ne "$SOURCE_COUNT" ]; then
        echo "ERROR: Count mismatch for images_${NUM}."
        echo "Source: $SOURCE_COUNT"
        echo "Output: $FINAL_COUNT"
        exit 1
    fi

    echo "Verified images_${NUM}: $FINAL_COUNT images."

    echo "Deleting full-resolution archive..."
    rm "$ARCHIVE"

    echo "Current full dataset size:"
    du -sh "$FULL"

    echo "Current quota:"
    quota -s

done

echo
echo "=================================================="
echo "All remaining archives processed."
echo "=================================================="

TOTAL=$(find "$FULL" -type f -name '*.png' | wc -l)

echo "Total PNG files: $TOTAL"
du -sh "$FULL"
quota -s
