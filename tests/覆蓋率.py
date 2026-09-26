#!/usr/bin/env python3
"""
项目全局测试与代码覆盖率自动化检查脚本
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    print(f"\n==========================================")
    print(f"🚀 正在执行: {description}")
    print(f"==========================================")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ 命令执行失败: {description}")
        return False
    return True


def main():
    # 确保当前目录在 PYTHONPATH 中
    os.environ["PYTHONPATH"] = "."

    # 1. 清理历史覆盖率缓存
    print("🧹 清理旧的覆盖率缓存...")
    subprocess.run("coverage erase", shell=True)

    # 2. 运行 pytest 并收集指定模块的覆盖率 (以 factory 目录为例)
    #    若需添加其他目录，可修改 --source 参数，例如: --source=factory,core,services
    test_cmd = "coverage run --source=factory -m pytest tests/"
    if not run_command(test_cmd, "运行单元测试并收集覆盖率"):
        sys.exit(1)

    # 3. 打印终端详细报告 (包含未覆盖的具体行号)
    print("\n" + "=" * 50)
    print("📊 代码覆盖率汇总报告 (Coverage Summary)")
    print("=" * 50)
    run_command("coverage report -m", "输出终端覆盖率报告")

    # 4. 自动生成 HTML 深度可视化报告
    print("\n" + "=" * 50)
    print("🌐 生成 HTML 可视化报告...")
    print("=" * 50)
    if run_command("coverage html", "生成 HTML 报告"):
        html_path = os.path.abspath("htmlcov/index.html")
        print(f"\n✅ HTML 报告生成完毕！可以在浏览器中打开以下路径查看逐行覆盖详情：\n👉 file://{html_path}\n")


if __name__ == "__main__":
    main()