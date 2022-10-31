#!/usr/bin/env bash

RES=generate-test.txt

cat $RES | grep '^T-[0-9]*' | sed -e 's/T-[0-9]*\s*//g' > test.ref
# cat $RES | grep '^D-[0-9]*' | sed -e 's/D-[0-9]*\s*[-0-9.]*\s*//g' > train.kd
cat $RES | grep '^S-[0-9]*' > test.src
cat $RES | grep '^D-[0-9]*' > train.kd
cat train.kd | wc -l
