#!/bin/sh

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
cmake.exe --build build --config Release
./move.sh
