#!/bin/sh
TS_TAG=TS-`date +%Y%m%d-%H%M`

echo Timestamp tag: ${TS_TAG}

cvs tag -c ${TS_TAG}
