#!/bin/sh

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$dir"
cd build && make -j8 && cd ..
mv -f build/*.so shadowbox
