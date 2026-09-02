# tests/test_rbac.py
import os
import logging
from factory.tool_factory import tool_factory, load_tools_from_yaml

# 设置日志格式
logging.basicConfig(level=logging.INFO)

def verify_rbac_rules():
    # 1. 显式装载 YAML 配置
    yaml_path = os.path.join(os.path.dirname(__file__), "tools.yaml")
    load_tools_from_yaml(yaml_path, tool_factory)
    
    # 假设 get_dataset_summary 的白名单配置为 ['admin', 'analyst']
    target_tool = "get_dataset_summary"
    
    print("\n========== 开始 RBAC 权限测试 ==========")
    
    # 测试场景 A：用 admin 角色获取高权工具
    admin_tool = tool_factory.get_tool(target_tool, user_role="admin")
    assert admin_tool is not None, "❌ 失败：Admin 应当拥有工具访问权限"
    print(f"✅ [Pass] Role 'admin' 成功获取工具: {target_tool}")
    
    # 测试场景 B：用普通 user 角色获取高权工具（预期拦截）
    user_tool = tool_factory.get_tool(target_tool, user_role="user")
    assert user_tool is None, "❌ 漏洞：User 越权获取到了高权工具！"
    print(f"✅ [Pass] Role 'user' 被成功阻断，无法获取工具: {target_tool}")
    
    # 测试场景 C：测试底层 _is_tool_visible
    all_tools = tool_factory._flat_tools
    if target_tool in all_tools:
        tool_obj = all_tools[target_tool]
        is_visible_user = tool_factory._is_tool_visible(tool_obj, user_role="user")
        is_visible_admin = tool_factory._is_tool_visible(tool_obj, user_role="admin")
        
        assert is_visible_user is False, "❌ 漏洞：工具对 user 依然可见！"
        assert is_visible_admin is True, "❌ 失败：工具对 admin 不可见！"
        print("✅ [Pass] 工具可见性过滤器 (_is_tool_visible) 校验正常")

    print("========================================\n")

if __name__ == "__main__":
    verify_rbac_rules()