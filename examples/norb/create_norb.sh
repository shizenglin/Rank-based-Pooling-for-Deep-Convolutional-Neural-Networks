#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/norb
DATA=data/norb
DBTYPE=leveldb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/norb-train-$DBTYPE $EXAMPLE/norb-test-$DBTYPE

./build/examples/norb/convert_norb_data.bin $DATA $EXAMPLE

