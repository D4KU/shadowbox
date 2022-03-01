#!/bin/sh

cd build && make -j8 && cd ..
mv -f build/*.so shadowbox
