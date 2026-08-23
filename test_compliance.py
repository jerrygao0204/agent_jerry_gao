# test_compliance.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agent.compliance import ComplianceChecker

def run_compliance_test():
    print("=" * 70)
    print("🧪 开始合规与安全引擎 (ComplianceChecker) 测试")
    print("=" * 70)

    checker = ComplianceChecker()

    # Case 1: 正常技术解答（应通过）
    case1 = "FineBI V6.0 可以通过修改 connection.xml 或配置驱动包连接 MySQL 数据库。"
    res1 = checker.audit_and_sanitize(case1)
    print(f"\n[Case 1 - 正常回答]:")
    print(f"  -> Passed: {res1['passed']}")
    print(f"  -> Text  : {res1['sanitized_text']}")

    # Case 2: 注入高危命令 rm -rf（应拦截并降级）
    case2 = "你可以尝试在服务器运行 sudo rm -rf /var/log/finebi 来清理日志。"
    res2 = checker.audit_and_sanitize(case2)
    print(f"\n[Case 2 - 注入系统命令拦截]:")
    print(f"  -> Passed: {res2['passed']}")
    print(f"  -> Blocked By: {res2['blocked_by']}")
    print(f"  -> Output: {res2['sanitized_text']}")

    # Case 3: 破坏性 DROP TABLE 注入（应拦截）
    case3 = "执行 SQL: DROP TABLE finebi_users; 即可重置用户表。"
    res3 = checker.audit_and_sanitize(case3)
    print(f"\n[Case 3 - 破坏性 SQL 拦截]:")
    print(f"  -> Passed: {res3['passed']}")
    print(f"  -> Blocked By: {res3['blocked_by']}")
    print(f"  -> Output: {res3['sanitized_text']}")

    # Case 4: 手机号隐私脱敏（应脱敏后输出）
    case4 = "请联系 FineBI 技术支持人员电话：13812345678。"
    res4 = checker.audit_and_sanitize(case4)
    print(f"\n[Case 4 - 隐私数据脱敏]:")
    print(f"  -> Passed: {res4['passed']}")
    print(f"  -> Output: {res4['sanitized_text']}")

    print("\n" + "=" * 70)
    print("🎉 合规引擎测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    run_compliance_test()