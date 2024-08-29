#!/usr/bin/env bash

bash ./build.sh

docker save surgvu_cat2 | gzip -c > surgtoolloc_det.tar.gz
