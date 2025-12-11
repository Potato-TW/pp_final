import re
import sys

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
        print(f"讀取文件時發生錯誤: {e}")
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
        
        # 提取line模式的投票時間 (Voting total)
        elif re.match(r'Voting total:', line) and current_mode == 'line':
            match = re.search(r'Voting total:\s*([\d.]+)\s*ms', line)
            if match and current_threads > 0:
                time_ms = float(match.group(1))
                results[current_mode][current_threads] = time_ms
        
        # 提取circle模式的投票時間 (Voting:)
        elif re.match(r'Voting:', line) and current_mode == 'circle':
            match = re.search(r'Voting:\s*([\d.]+)\s*ms', line)
            if match and current_threads > 0:
                time_ms = float(match.group(1))
                results[current_mode][current_threads] = time_ms
    
    return results

def format_results(results):
    """
    格式化結果為要求的表格格式
    
    Args:
        results (dict): 解析後的時間數據
        
    Returns:
        str: 格式化後的表格字符串
    """
    if not results or not results['line'] or not results['circle']:
        return "錯誤: 無法從文件中提取有效的時間數據"
    
    output = ["#threads line(ms) circle(ms)"]
    
    # 獲取所有thread數量（1-16）
    threads = sorted(set(results['line'].keys()) | set(results['circle'].keys()))
    
    for t in threads:
        line_time = results['line'].get(t, 'N/A')
        circle_time = results['circle'].get(t, 'N/A')
        output.append(f"{t} {line_time} {circle_time}")
    
    return '\n'.join(output)

def main():
    """
    主函數：處理命令行參數並執行解析
    """
    if len(sys.argv) != 2:
        print("使用方法: python parse_hough_timing.py <日誌文件名>")
        print("示例: python parse_hough_timing.py hough_log.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"正在解析文件: {filename}")
    
    # 解析文件
    results = parse_hough_timing_from_file(filename)
    
    if results:
        # 格式化並輸出結果
        formatted_output = format_results(results)
        print("\n解析結果:")
        print(formatted_output)
        
        # 保存結果到文件（可選）
        output_filename = f"parsed_{filename}"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"\n結果已保存到: {output_filename}")
        except Exception as e:
            print(f"保存結果文件時發生錯誤: {e}")
    else:
        print("解析失敗，請檢查文件格式")

if __name__ == "__main__":
    main()
