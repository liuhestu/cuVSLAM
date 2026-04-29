import subprocess
import time
import psutil
import statistics
import sys

def get_gpu_utilization():
    """通过 nvidia-smi 获取当前 GPU 整体占用率"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        return float(result.strip().split('\n')[0])
    except Exception:
        return 0.0

def main():
    # 检查是否传入了要运行的文件名
    if len(sys.argv) < 2:
        print("用法: python3 monitor.py <你要单独运行的脚本.py>")
        sys.exit(1)

    target_script = sys.argv[1]

    print(f"=== 准备测试单个文件: {target_script} ===")
    print("正在收集系统基准负载(耗时3秒，请不要操作电脑)...")

    base_cpus = []
    base_gpus = []
    # 采样3秒的系统空闲状态
    for _ in range(3):
        base_cpus.append(psutil.cpu_percent(interval=1))
        base_gpus.append(get_gpu_utilization())

    base_cpu_avg = statistics.mean(base_cpus)
    base_gpu_avg = statistics.mean(base_gpus)
    print(f"【基准】当前系统总CPU占用: {base_cpu_avg:.1f}%, GPU占用: {base_gpu_avg:.1f}%\n")

    print(f"=== 正在启动并监控 {target_script} ===")
    print("提示: 随时可以在终端按 Ctrl+C 结束测试并查看统计结果\n")

    # 启动你要测试的脚本
    process = subprocess.Popen([sys.executable, target_script])

    # 初始化记录容器
    run_cpus = []
    run_gpus = []

    # 先初始化一次 psutil 的非阻塞计数器
    psutil.cpu_percent(interval=None)

    try:
        while process.poll() is None:          # 当目标进程还在运行时
            # 采集瞬时 CPU 利用率（相对上一次调用）
            cpu = psutil.cpu_percent(interval=None)
            gpu = get_gpu_utilization()

            run_cpus.append(cpu)
            run_gpus.append(gpu)

            time.sleep(0.5)                   # 采样间隔，可调
    except KeyboardInterrupt:
        print("\n手动中止监控，正在生成报告...")
        process.terminate()
        process.wait()

    # 计算运行期间的平均值与最大值
    if run_cpus:
        avg_cpu = statistics.mean(run_cpus)
        max_cpu = max(run_cpus)
        avg_gpu = statistics.mean(run_gpus)
        max_gpu = max(run_gpus)
    else:
        avg_cpu = max_cpu = avg_gpu = max_gpu = 0.0

    print("\n========== 监控统计 ==========")
    print(f"基准 CPU : {base_cpu_avg:.1f}%")
    print(f"运行期间 CPU 均值: {avg_cpu:.1f}% , 峰值: {max_cpu:.1f}%")
    print(f"平均CPU占用增加: {avg_cpu - base_cpu_avg:.1f}%")
    print(f"基准 GPU : {base_gpu_avg:.1f}%")
    print(f"运行期间 GPU 均值: {avg_gpu:.1f}% , 峰值: {max_gpu:.1f}%")
    print(f"平均GPU占用增加: {avg_gpu - base_gpu_avg:.1f}%")
    print(f"采样点数: {len(run_cpus)}")

if __name__ == "__main__":
    main()