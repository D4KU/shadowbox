#!/bin/sh

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"

mv -f build/*.so shadowbox
mv -f build/Release/*.pyd shadowbox
./sync.sh
