#!/bin/bash

storage_account=fusong4dataset
SAS_STR="sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2027-04-02T14:37:20Z&st=2022-04-02T06:37:20Z&spr=https&sig="

SSSSSS="9V92v9a7xITRAspz06jJr%2F1q%2BPg3qXzwpmgIzTJsXZk%3D"
dataset="dataset/R2P/Saturn-finetune"

##############################################################################

HERE="$(dirname "$(readlink -f "$0")")"
HERE="$(dirname "$HERE")"

AZCOPY=$HERE/third_party/azcopy/azcopy

if [ ! -e $AZCOPY ]; then
  outdir=$HERE/third_party/azcopy
  mkdir -p $outdir
  wget -O $outdir/azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && \
    tar -xf $outdir/azcopy_v10.tar.gz -C $outdir --strip-components=1
fi

SAS_CTN="https://$storage_account.blob.core.windows.net"
$AZCOPY cp "${SAS_CTN}/$dataset/?${SAS_STR}${SSSSSS}" $HERE/dataset --recursive=true
