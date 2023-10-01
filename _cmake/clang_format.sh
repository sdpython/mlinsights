#!/bin/bash
clear
echo "--ruff--"
ruff .
echo "--cython-lint--"
cython-lint .
echo "--clang-format--"
find mlinsights -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cu" \) | while read f; do
    echo "clang-format -i $f";
    clang-format -i $f;
done
echo "--cmake-lint--"
find _cmake -type f \( -name "*.cmake" -o -name "*.txt" \) | while read f; do
    echo "cmake-lint $f --line-width=88 --disabled-codes C0103 C0113";
    cmake-lint $f --line-width=88 --disabled-codes C0103 C0113;
done
