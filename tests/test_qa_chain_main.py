# tests/test_qa_chain_main.py
import subprocess
import sys
import os

def test_qa_chain_main_help():
    """Smoke test: 執行 qa_chain.py 的 CLI help"""
    script_path = os.path.join(os.path.dirname(__file__), "..", "generator", "qa_chain.py")
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )
    # 確保正常退出
    assert result.returncode == 0
    # 輸出中應該包含 usage/help 提示
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()
