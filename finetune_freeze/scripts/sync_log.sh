#!/bin/bash

HERE="$(dirname "$(readlink -f "$0")")"

exp_name=$($HERE/show_experiment_name.py)

set -x

amlt log $exp_name
