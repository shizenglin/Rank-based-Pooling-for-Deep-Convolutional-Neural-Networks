#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar100
DATA=data/cifar100
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar100_train_$DBTYPE $EXAMPLE/cifar100_test_$DBTYPE

./build/examples/cifar100/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar100_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
