#!/bin/bash

make clean ; make

for mode in line circle; do
    log_file="log_${mode}.txt"
    : > "$log_file"   # 清空舊檔案

    for i in {1..16}; do
        echo "Threads: $i" >> "$log_file"
        ./hough_star_pthreads.out ../star.png --mode "$mode" -t "$i" >> "$log_file"
        echo "" >> "$log_file"     # 印空行
    done
done
