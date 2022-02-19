#!/usr/bin/env bash

# This script allows to quickly change into the latest log dir of a specific kind.
# Should be executed from project root folder.
#
# Go to latest log dir from 'runs' folder:
# cd $(logs/latest logs/runs)
#
# Go to latest log dir from 'experiments' folder:
# cd $(logs/latest logs/experiments)


set -e
set -o pipefail

find "$1" -maxdepth 2 -type d | sort -r | head -n 1
