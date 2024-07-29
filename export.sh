#!/usr/bin/env bash

./build.sh

docker save surgvu_cat2 | gzip -c > surgtoolloc_det.tar.gz
