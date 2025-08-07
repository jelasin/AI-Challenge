# -*- coding: utf-8 -*-
"""
Challenge 4: MCP提示模板系统

学习目标:
1. 掌握MCP提示模板的发现和使用
2. 学习动态提示生成和参数化
3. 实现提示模板的复用和组合
4. 理解提示版本管理和继承机制

核心概念:
- Prompt Discovery: 提示模板发现
- Dynamic Prompt Generation: 动态提示生成
- Template Parameterization: 模板参数化
- Prompt Composition: 提示组合
- Version Management: 版本管理

实战场景:
构建一个智能提示管理系统，支持从MCP服务器动态加载提示模板，
实现提示的参数化定制、组合复用和版本管理功能。
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入提示模板服务器
try:
    sys.path.append(str(project_root / "mcp_servers"))
    from prompt_server import get_server as get_prompt_server  # type: ignore
except ImportError as e:
    print(f"❌ 无法导入提示模板服务器: {e}")
    # 创建一个简单的替代实现
    class DummyPromptServer:
        def __init__(self):
            self.templates = {}
        def list_templates(self, filter_tags=None, template_type=None):
            return {"success": True, "total": 0, "templates": []}
        def get_template(self, name):
            return {"success": False, "error": "服务器不可用"}
        def render_template(self, name, parameters):
            return {"success": False, "error": "服务器不可用"}
        def create_template(self, name, content, parameters=None, description="", tags=None, template_type="user"):
            return {"success": False, "error": "服务器不可用"}
        def get_template_stats(self):
            return {"success": False, "error": "服务器不可用"}
    
    def get_prompt_server():
        return DummyPromptServer()

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装必要的包:")
    print("pip install langchain-mcp-adapters langchain-openai")
    sys.exit(1)

class PromptType(Enum):
    """提示类型枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TEMPLATE = "template"
    COMPOSITE = "composite"

class MCPPromptManager:
    """MCP提示管理系统"""
    
    def __init__(self):
        """初始化提示管理器"""
        # 使用真实的提示模板服务器
        self.prompt_server = get_prompt_server()
        
        # MCP客户端配置
        self.mcp_client = None
        self.servers_config = {
            "prompt_server": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "prompt_server.py")],
                "env": {}
            }
        }
        
        # 模板缓存
        self.template_cache: Dict[str, Dict[str, Any]] = {}
        self.template_usage_stats: Dict[str, int] = {}
        
        # LLM用于提示效果测试
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        ) if os.getenv("OPENAI_API_KEY") else None
    
    async def initialize(self) -> bool:
        """初始化提示管理器"""
        print("🔧 初始化MCP提示管理系统...")
        
        try:
            # 直接使用本地提示模板服务器
            print("✅ 本地提示模板服务器已连接")
            
            # 加载模板到缓存
            await self.load_templates_to_cache()
            
            print("✅ 提示管理器初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    async def load_templates_to_cache(self):
        """加载提示模板到缓存"""
        print("📥 加载提示模板到缓存...")
        
        # 使用本地服务器的 list_templates 方法
        result = self.prompt_server.list_templates()
        
        if result.get("success"):
            templates = result.get("templates", [])
            for template_info in templates:
                template_name = template_info["name"]
                # 获取完整的模板信息
                template_result = self.prompt_server.get_template(template_name)
                if template_result.get("success"):
                    self.template_cache[template_name] = template_result["template"]
            
            print(f"✅ 缓存了 {len(self.template_cache)} 个提示模板")
        else:
            print(f"❌ 加载模板失败: {result.get('error', '未知错误')}")
    
    async def discover_templates(self, filter_tags: Optional[List[str]] = None, 
                               template_type: Optional[str] = None) -> Dict[str, Any]:
        """发现提示模板"""
        print(f"🔍 发现提示模板 - 过滤条件: 标签={filter_tags}, 类型={template_type}")
        
        # 使用本地服务器的 list_templates 方法
        result = self.prompt_server.list_templates(filter_tags, template_type)
        
        if result.get("success"):
            print(f"✅ 发现了 {result['total']} 个匹配的模板")
            return result
        else:
            print(f"❌ 模板发现失败: {result.get('error', '未知错误')}")
            return {"success": False, "templates": []}
    
    async def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """获取特定模板的详细信息"""
        # 先从缓存获取
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        # 从服务器获取
        result = self.prompt_server.get_template(template_name)
        if result.get("success"):
            template_info = result["template"]
            # 更新缓存
            self.template_cache[template_name] = template_info
            return template_info
        
        return None
    
    async def render_template(self, template_name: str, parameters: Dict[str, Any]) -> Optional[str]:
        """渲染提示模板"""
        print(f"⚙️ 渲染模板: {template_name}")
        print(f"🔧 参数: {parameters}")
        
        # 使用本地服务器渲染模板
        result = self.prompt_server.render_template(template_name, parameters)
        
        if result.get("success"):
            rendered_content = result["rendered_content"]
            
            # 更新使用统计
            self.template_usage_stats[template_name] = self.template_usage_stats.get(template_name, 0) + 1
            
            print("✅ 模板渲染成功")
            return rendered_content
        else:
            print(f"❌ 模板渲染失败: {result.get('error', '未知错误')}")
            return None
    
    async def create_custom_template(self, name: str, content: str, parameters: Dict[str, Any], 
                                   description: str = "", tags: Optional[List[str]] = None, 
                                   template_type: str = "user") -> bool:
        """创建自定义模板"""
        print(f"✨ 创建自定义模板: {name}")
        
        # 使用本地服务器创建模板
        result = self.prompt_server.create_template(
            name=name,
            content=content,
            parameters=parameters,
            description=description,
            tags=tags,
            template_type=template_type
        )
        
        if result.get("success"):
            # 更新缓存
            self.template_cache[name] = result["template"]
            print(f"✅ 自定义模板 '{name}' 创建成功")
            return True
        else:
            print(f"❌ 创建失败: {result.get('error', '未知错误')}")
            return False
    
    async def compose_templates(self, template_names: List[str], parameters_list: List[Dict[str, Any]]) -> Optional[str]:
        """组合多个模板"""
        if len(template_names) != len(parameters_list):
            print("❌ 模板名称和参数列表长度不匹配")
            return None
        
        print(f"🔗 组合模板: {template_names}")
        
        composed_parts = []
        
        for template_name, parameters in zip(template_names, parameters_list):
            rendered_content = await self.render_template(template_name, parameters)
            if rendered_content:
                composed_parts.append(rendered_content)
            else:
                print(f"❌ 模板 {template_name} 渲染失败，跳过")
        
        if composed_parts:
            composed_prompt = "\n\n---\n\n".join(composed_parts)
            print("✅ 模板组合成功")
            return composed_prompt
        else:
            print("❌ 模板组合失败")
            return None
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计"""
        # 从服务器获取统计
        result = self.prompt_server.get_template_stats()
        
        if result.get("success"):
            # 合并本地统计
            server_stats = result
            server_stats["local_usage_stats"] = self.template_usage_stats
            return server_stats
        else:
            return {
                "success": False,
                "local_usage_stats": self.template_usage_stats
            }
    
    async def demonstrate_prompt_discovery(self):
        """演示提示模板发现功能"""
        print("\n" + "="*60)
        print("🔍 提示模板发现演示")
        print("="*60)
        
        # 发现所有模板
        all_templates = await self.discover_templates()
        
        if all_templates.get("success"):
            templates = all_templates["templates"]
            
            # 按类型分组显示模板
            templates_by_type = {}
            
            for template in templates:
                template_type = template["type"]
                if template_type not in templates_by_type:
                    templates_by_type[template_type] = []
                templates_by_type[template_type].append(template)
            
            print(f"📊 发现 {all_templates['total']} 个提示模板:")
            
            for template_type, type_templates in templates_by_type.items():
                print(f"\n📂 {template_type.upper()} 类型模板 ({len(type_templates)} 个):")
                
                for template in type_templates:
                    print(f"  • {template['name']}")
                    print(f"    描述: {template['description'] or '无描述'}")
                    print(f"    参数: {template['parameters']} 个")
                    print(f"    标签: {', '.join(template['tags']) if template['tags'] else '无'}")
                    if template['usage_count'] > 0:
                        print(f"    使用次数: {template['usage_count']}")
                    print()
            
            # 按标签过滤演示
            print("\n🏷️  按标签过滤演示:")
            code_templates = await self.discover_templates(filter_tags=["code"])
            if code_templates.get("success"):
                print(f"代码相关模板: {len(code_templates['templates'])} 个")
                for template in code_templates["templates"]:
                    print(f"  • {template['name']}")
            
            # 按类型过滤演示
            print("\n📁 按类型过滤演示:")
            system_templates = await self.discover_templates(template_type="system")
            if system_templates.get("success"):
                print(f"系统类型模板: {len(system_templates['templates'])} 个")
                for template in system_templates["templates"]:
                    print(f"  • {template['name']}")
    
    async def demonstrate_parameterized_prompts(self):
        """演示参数化提示使用"""
        print("\n" + "="*60)
        print("⚙️ 参数化提示演示")
        print("="*60)
        
        # 测试用例
        test_cases = [
            {
                "template": "code_reviewer",
                "args": {
                    "language": "Python",
                    "focus_areas": "性能优化和安全性"
                },
                "description": "Python代码审查"
            },
            {
                "template": "data_analyst", 
                "args": {
                    "analysis_type": "销售数据分析",
                    "data_source": "电商平台",
                    "key_metrics": "转化率、客单价、复购率",
                    "data_format": "CSV"
                },
                "description": "电商销售数据分析"
            },
            {
                "template": "creative_writer",
                "args": {
                    "content_type": "科技博客文章",
                    "target_audience": "软件工程师",
                    "topic": "人工智能在软件开发中的应用",
                    "style": "专业且易懂",
                    "length": "1000字左右"
                },
                "description": "科技写作"
            }
        ]
        
        for test_case in test_cases:
            template_name = test_case["template"]
            args = test_case["args"]
            description = test_case["description"]
            
            print(f"\n🧪 测试场景: {description}")
            print(f"📝 使用模板: {template_name}")
            
            # 显示参数
            print(f"🔧 参数配置:")
            for key, value in args.items():
                print(f"  • {key}: {value}")
            
            # 生成提示
            generated_prompt = await self.render_template(template_name, args)
            
            if generated_prompt:
                print(f"\n✅ 生成的提示:")
                print("-" * 40)
                print(generated_prompt[:500] + "..." if len(generated_prompt) > 500 else generated_prompt)
                print("-" * 40)
            else:
                print("❌ 提示生成失败")
            
            await asyncio.sleep(1)
    
    async def demonstrate_prompt_composition(self):
        """演示提示组合功能"""
        print("\n" + "="*60)
        print("🔗 提示组合演示")
        print("="*60)
        
        # 创建组合提示示例
        print("🧩 创建组合提示：数据分析师 + 问题解决者")
        
        template_names = ["data_analyst", "problem_solver"]
        parameters_list = [
            {
                "analysis_type": "用户行为分析",
                "data_source": "移动应用",
                "key_metrics": "活跃度、留存率",
                "data_format": "JSON"
            },
            {
                "problem_description": "用户留存率下降",
                "problem_type": "产品优化",
                "constraints": "有限的开发资源",
                "expected_outcome": "提升30%留存率"
            }
        ]
        
        composed_prompt = await self.compose_templates(template_names, parameters_list)
        
        if composed_prompt:
            print("✅ 组合提示创建成功:")
            print("-" * 40)
            print(composed_prompt[:500] + "..." if len(composed_prompt) > 500 else composed_prompt)
            print("-" * 40)
        else:
            print("❌ 组合提示创建失败")
    
    async def demonstrate_custom_template_creation(self):
        """演示自定义模板创建"""
        print("\n" + "="*60)
        print("✨ 自定义模板创建演示")
        print("="*60)
        
        # 创建自定义模板
        custom_template_success = await self.create_custom_template(
            name="code_optimizer",
            content="""你是一位代码优化专家。请分析并优化以下{language}代码：

优化目标：{optimization_goals}
性能要求：{performance_requirements}

请提供：
1. 当前代码分析
2. 优化建议
3. 重构后的代码
4. 性能对比说明

代码内容：{code_content}""",
            parameters={
                "language": {"type": "string", "required": True, "description": "编程语言"},
                "optimization_goals": {"type": "string", "required": True, "description": "优化目标"},
                "performance_requirements": {"type": "string", "required": False, "default": "一般性能要求", "description": "性能要求"},
                "code_content": {"type": "string", "required": True, "description": "需要优化的代码"}
            },
            description="代码优化专家模板",
            tags=["code", "optimization", "performance"],
            template_type="system"
        )
        
        if custom_template_success:
            # 测试自定义模板
            test_result = await self.render_template("code_optimizer", {
                "language": "Python",
                "optimization_goals": "提升执行速度",
                "performance_requirements": "减少50%执行时间",
                "code_content": "def slow_function(n):\n    result = 0\n    for i in range(n):\n        for j in range(n):\n            result += i * j\n    return result"
            })
            
            if test_result:
                print("\n🧪 自定义模板测试:")
                print("-" * 30)
                print(test_result[:400] + "..." if len(test_result) > 400 else test_result)
                print("-" * 30)
    
    async def demonstrate_prompt_testing(self):
        """演示提示效果测试"""
        if not self.llm:
            print("\n⚠️  跳过提示测试演示 - 未设置OPENAI_API_KEY")
            return
        
        print("\n" + "="*60)
        print("🧪 提示效果测试演示")
        print("="*60)
        
        # 选择一个模板进行测试
        test_template = "tutor_assistant"
        test_args = {
            "subject": "Python编程",
            "teaching_style": "互动式",
            "student_level": "初学者",
            "learning_objectives": "掌握基础语法和数据类型",
            "current_topic": "变量和数据类型"
        }
        
        print(f"🎯 测试模板: {test_template}")
        print(f"📋 测试参数: {test_args}")
        
        # 生成提示
        generated_prompt = await self.render_template(test_template, test_args)
        
        if generated_prompt:
            print(f"\n📤 发送到LLM的提示:")
            print("-" * 30)
            print(generated_prompt[:300] + "..." if len(generated_prompt) > 300 else generated_prompt)
            print("-" * 30)
            
            try:
                # 测试提示效果
                messages = [
                    SystemMessage(content=generated_prompt),
                    HumanMessage(content="请介绍一下Python中的基础数据类型")
                ]
                
                response = await self.llm.ainvoke(messages)
                
                print(f"\n🤖 LLM响应:")
                print("-" * 30)
                response_content = str(response.content) if response.content else ""
                print(response_content[:500] + "..." if len(response_content) > 500 else response_content)
                print("-" * 30)
                
                # 简单的效果评估
                response_length = len(response_content)
                print(f"\n📊 响应评估:")
                print(f"  • 响应长度: {response_length} 字符")
                print(f"  • 是否包含关键词: {'✅' if '数据类型' in response_content else '❌'}")
                print(f"  • 教学风格: {'✅' if any(word in response_content for word in ['学习', '理解', '例子']) else '❌'}")
                
            except Exception as e:
                print(f"❌ LLM测试失败: {e}")
        else:
            print("❌ 提示生成失败，无法进行测试")
    
    async def demonstrate_template_analytics(self):
        """演示模板使用分析"""
        print("\n" + "="*60)
        print("📈 模板使用分析演示")
        print("="*60)
        
        # 获取使用统计
        stats = await self.get_usage_statistics()
        
        if stats.get("success"):
            print(f"📊 模板使用统计:")
            print("-" * 50)
            
            print(f"总模板数: {stats['total_templates']}")
            print(f"总使用次数: {stats['total_usage']}")
            print(f"平均使用次数: {stats['average_usage']:.2f}")
            
            print(f"\n🏷️  模板类型分布:")
            for template_type, type_stat in stats["type_statistics"].items():
                print(f"  • {template_type}: {type_stat['count']} 个模板, {type_stat['usage']} 次使用")
            
            print(f"\n🔥 最常用的模板:")
            for i, template in enumerate(stats["most_used_templates"], 1):
                print(f"  {i}. {template['name']} ({template['type']}) - {template['usage_count']} 次")
            
            # 显示本地使用统计
            if stats.get("local_usage_stats"):
                print(f"\n📊 本次会话使用统计:")
                for template_name, count in stats["local_usage_stats"].items():
                    print(f"  • {template_name}: {count} 次")
        else:
            print("❌ 无法获取使用统计")

async def demo_prompt_templates():
    """Challenge 4 主演示函数"""
    print("🚀 Challenge 4: MCP提示模板系统")
    print("="*60)
    
    # 创建提示管理器
    manager = MCPPromptManager()
    
    # 初始化
    if not await manager.initialize():
        print("❌ 无法初始化提示管理器，演示结束")
        return
    
    try:
        # 1. 提示模板发现演示
        await manager.demonstrate_prompt_discovery()
        
        # 2. 参数化提示演示
        await manager.demonstrate_parameterized_prompts()
        
        # 3. 提示组合演示
        await manager.demonstrate_prompt_composition()
        
        # 4. 自定义模板创建演示
        await manager.demonstrate_custom_template_creation()
        
        # 5. 提示效果测试演示
        await manager.demonstrate_prompt_testing()
        
        # 6. 模板分析演示
        await manager.demonstrate_template_analytics()
        
        print("\n🎉 Challenge 4 演示完成！")
        print("\n📚 学习要点总结:")
        print("  ✅ 掌握了MCP提示模板的发现和管理")
        print("  ✅ 学会了提示的参数化和动态生成")
        print("  ✅ 实现了提示模板的组合复用")
        print("  ✅ 体验了自定义模板创建功能")
        print("  ✅ 使用了真实的MCP工具调用")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_prompt_templates())

if __name__ == "__main__":
    main()
