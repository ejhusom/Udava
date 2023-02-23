#!/bin/bash
# ===================================================================
# File:     build_and_run_udava_docker.sh
# Author:   Erik Johannes Husom
# Created:
# -------------------------------------------------------------------
# Description: Build and run Udava Docker container.
# ===================================================================

docker build -t udava -f Dockerfile .
docker run -p 5000:5000 -v $(pwd)/assets:/usr/Udava/assets -v $(pwd)/.dvc:/usr/Udava/.dvc udava
