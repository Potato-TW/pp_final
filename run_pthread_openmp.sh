#!/bin/bash

cd pic/
python3 generate_img.py
cd ../

cd pthread/
make clean ; make
cd ../

cd openMP/
make clean ; make
cd ../

# mode 可以是 line / circle
modes=("line" "circle")

# 圖片列表
images=(./pic/*.png)  # 假設 script 在 ../

# 目錄列表
dirs=("pthread" "openMP")

for dir in "${dirs[@]}"; do
    # 找目錄下所有 .out 執行檔
    targets=("$dir"/*.out)

    for tgt in "${targets[@]}"; do
        tgt_name=$(basename "$tgt")

        for img in "${images[@]}"; do
            img_base=$(basename "$img" .png)
            # log 放在對應目錄下
            log_file="$dir/log_${tgt_name}_${img_base}.txt"
            : > "$log_file"  # 清空舊檔案

            echo "=== Image: $img ===" >> "$log_file"

            for mode in "${modes[@]}"; do
                echo "--- Mode: $mode ---" >> "$log_file"

                for i in {1..16}; do
                    echo "Threads: $i" >> "$log_file"
                    ./"$tgt" "$img" --mode "$mode" -t "$i" >> "$log_file" 2>&1
                    echo "" >> "$log_file"
                done
            done
        done
    done
done

echo "All runs finished. Logs are saved in pthread/ and openMP/"
