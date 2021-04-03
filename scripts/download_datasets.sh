#!/bin/bash

# Download datasets
DATA_BASE_PATH="./data"
mkdir -p "${DATA_BASE_PATH}"

# EgoHands
wget "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip" -O "${DATA_BASE_PATH}/egohands_data.zip"
unzip "${DATA_BASE_PATH}/egohands_data.zip" -d "${DATA_BASE_PATH}/egohands_data"
rm "${DATA_BASE_PATH}/egohands_data.zip"

# EgoYouTubeHands (EYTH)
gdown "https://drive.google.com/uc?id=1EwjJx-V-Gq7NZtfiT6LZPLGXD2HN--qT" -O "${DATA_BASE_PATH}/eyth_dataset.tar.gz"
tar -xvf "${DATA_BASE_PATH}/eyth_dataset.tar.gz" -C "${DATA_BASE_PATH}"
rm "${DATA_BASE_PATH}/eyth_dataset.tar.gz"
rename 's/$/\.png/' ${DATA_BASE_PATH}/eyth_dataset/masks/vid4/*
rename 's/$/\.png/' ${DATA_BASE_PATH}/eyth_dataset/masks/vid9/*

# GTEA
wget "https://www.dropbox.com/s/k2g06apgx1u8p17/hand2K_dataset.zip" -O "${DATA_BASE_PATH}/hand2K_dataset.zip"
unzip "${DATA_BASE_PATH}/hand2K_dataset.zip" -d "${DATA_BASE_PATH}/hand2K_dataset"
rm "${DATA_BASE_PATH}/hand2K_dataset.zip"

# HandOverFace (HOF)
gdown "https://drive.google.com/uc?id=1hHUvINGICvOGcaDgA5zMbzAIUv7ewDd3" -O "${DATA_BASE_PATH}/hand_over_face_corrected.tar.gz"
tar -xvf "${DATA_BASE_PATH}/hand_over_face_corrected.tar.gz" -C "${DATA_BASE_PATH}"
rm "${DATA_BASE_PATH}/hand_over_face_corrected.tar.gz"
