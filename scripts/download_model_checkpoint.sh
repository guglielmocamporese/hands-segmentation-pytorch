#!/bin/bash

# Download checkpoints
CKPT_OUTPATH=./checkpoint
mkdir -p $CKPT_OUTPATH

echo "##################################################"
echo "# Downloading model checkpoints..."
echo "##################################################"
echo
gdown --output "${CKPT_OUTPATH}/checkpoint.ckpt"            "1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg"
gdown --output "${CKPT_OUTPATH}/checkpoint-grayscale.ckpt"  "1dJTPikk1V4No-4vfq1269IFEfbDZvrV6"
echo
echo "##################################################"
echo "# All done! Model checkpoint downloaded in ${CKPT_OUTPATH}"
echo "##################################################"
