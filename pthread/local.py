import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_hough_timing_from_file(filename):
    """
    從Hough變換日誌文件中解析多線程性能數據
    
    Args:
        filename (str): 日誌文件路徑
        
    Returns:
        dict: 包含line和circle模式時間數據的字典
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"錯誤: 找不到文件 {filename}")
        return None
    except Exception as e:
        print(f"讀取文件 {filename} 時發生錯誤: {e}")
        return None
    
    results = {'line': {}, 'circle': {}}
    current_mode = None
    current_threads = 0
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 檢測模式切換
        if 'Mode:' in line:
            if 'line' in line:
                current_mode = 'line'
            elif 'circle' in line:
                current_mode = 'circle'
        
        # 檢測thread數量
        elif re.match(r'Threads:\s*\d+', line):
            match = re.search(r'Threads:\s*(\d+)', line)
            if match:
                current_threads = int(match.group(1))
        
        # 提取投票時間 (Voting:)
        elif re.match(r'Voting:', line):
            match = re.search(r'Voting:\s*([\d.]+)\s*ms', line)
            if match and current_mode and current_threads > 0:
                time_ms = float(match.group(1))
                results[current_mode][current_threads] = time_ms
    
    return results

def extract_legend_label(filename):
    """
    從文件名中提取圖例標籤
    例如: log_hough_avx_pthreads.out_star_full.txt -> avx_full
    """
    # 提取文件名（去除路徑）
    basename = filename.split('/')[-1] if '/' in filename else filename
    
    # 使用正則表達式提取 avx_full 部分
    match = re.search(r'avx_([a-zA-Z0-9_]+)', basename)
    if match:
        return f"avx_{match.group(1)}"
    
    # 如果沒有匹配，返回文件名（不含擴展名）
    return basename.split('.')[0]

def create_line_plot(data_dict, output_filename):
    """
    創建line模式的折線圖
    
    Args:
        data_dict: 包含多個文件數據的字典
        output_filename: 輸出文件名
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '*']
    
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'line' in data and data['line']:
            threads = sorted(data['line'].keys())
            times = [data['line'][t] for t in threads]
            
            # 從文件名提取圖例標籤
            label = extract_legend_label(filename)
            plt.plot(threads, times, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    markersize=8,
                    label=label)
    
    plt.xlabel('Threads', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.title('Hough Line Detection Performance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(range(1, 17))
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Line圖表已保存到: {output_filename}")

def create_circle_plot(data_dict, output_filename):
    """
    創建circle模式的折線圖
    
    Args:
        data_dict: 包含極個文件數據的字典
        output_filename: 輸出文件名
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '*']
    
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'circle' in data and data['circle']:
            threads = sorted(data['circle'].keys())
            times = [data['circle'][t] for t in threads]
            
            # 從文件名提取圖例標籤
            label = extract_legend_label(filename)
            plt.plot(threads, times, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    markersize=8,
                    label=label)
    
    plt.xlabel('Threads', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.title('Hough Circle Detection Performance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(range(1, 17))
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Circle圖表已保存到: {output_filename}")

def main():
    """
    主函數：處理命令行參數並執行解析
    """
    if len(sys.argv) < 2:
        print("使用方法: python plot_hough_performance.py <日誌文件1> <日誌文件2> ...")
        print("示例: python plot_hough_performance.py log_hough_avx_pthreads.out_star_full.txt ...")
        sys.exit(1)
    
    filenames = sys.argv[1:]
    
    # 解析所有文件
    all_data = {}
    for filename in filenames:
        print(f"正在解析文件: {filename}")
        data = parse_hough_timing_from_file(filename)
        if data:
            all_data[filename] = data
            print(f"  成功解析 - 圖例標籤: {extract_legend_label(filename)}")
        else:
            print(f"  跳過文件: {filename}")
    
    if not all_data:
        print("沒有成功解析任何文件")
        return
    
    # 創建圖表
    create_line_plot(all_data, "hough_line_performance.png")
    create_circle_plot(all_data, "hough_circle_performance.png")
    
    # 輸出數據摘要
    print("\n數據摘要:")
    for filename, data in all_data.items():
        label = extract_legend_label(filename)
        print(f"\n{label}:")
        if 'line' in data:
            best_thread = min(data['line'], key=data['line'].get)
            best_time = data['line'][best_thread]
            print(f"  Line - 最佳時間: {best_time:.2f}ms (threads={best_thread})")
        if 'circle' in data:
            best_thread = min(data['circle'], key=data['circle'].get)
            best_time = data['circle'][best_thread]
            print(f"  Circle - 最佳時間: {best_time:.2f}ms (threads={best_thread})")

if __name__ == "__main__":
    main()
