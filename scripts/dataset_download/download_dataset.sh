#!/bin/bash
DATASET_DIR="/home/ubuntu/joule/datasets/alexandria"
mkdir -p $DATASET_DIR

curl -L -o $DATASET_DIR/alexandria_ps_000.json.bz2 "https://archive.materialscloud.org/record/file?record_id=1755&filename=alexandria_ps_000.json.bz2"
curl -L -o $DATASET_DIR/alexandria_ps_001.json.bz2 "https://archive.materialscloud.org/record/file?record_id=1755&filename=alexandria_ps_001.json.bz2"
curl -L -o $DATASET_DIR/alexandria_ps_002.json.bz2 "https://archive.materialscloud.org/record/file?record_id=1755&filename=alexandria_ps_002.json.bz2"
curl -L -o $DATASET_DIR/alexandria_ps_003.json.bz2 "https://archive.materialscloud.org/record/file?record_id=1755&filename=alexandria_ps_003.json.bz2"
curl -L -o $DATASET_DIR/alexandria_ps_004.json.bz2 "https://archive.materialscloud.org/record/file?record_id=1755&filename=alexandria_ps_004.json.bz2"
