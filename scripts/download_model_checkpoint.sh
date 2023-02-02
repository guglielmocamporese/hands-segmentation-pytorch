#!/bin/bash

# Utility function
function download_ckpt {
    FILEID=$1
    OUTNAME=$2

    out="$(wget \
        -q \
        --save-cookies /tmp/cookies.txt \
        --keep-session-cookies \
        --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" \
        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')"


    wget \
        -q --show-progress \
        --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=${out}&id=$FILEID" -O $OUTNAME && rm -rf /tmp/cookies.txt
}

# Download checkpoints
CKPT_OUTPATH=./checkpoint
mkdir -p $CKPT_OUTPATH

echo "##################################################"
echo "# Downloading model checkpoints..."
echo "##################################################"
echo
download_ckpt "1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg" "${CKPT_OUTPATH}/checkpoint.ckpt"
download_ckpt "1dJTPikk1V4No-4vfq1269IFEfbDZvrV6" "${CKPT_OUTPATH}/checkpoint-grayscale.ckpt"
echo
echo "##################################################"
echo "# All done! Model checkpoint downloaded in ${CKPT_OUTPATH}"
echo "##################################################"
