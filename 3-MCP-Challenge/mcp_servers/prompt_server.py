#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 提示模板服务器

提供提示模板的创建、管理、发现和使用功能。
支持参数化提示、模板组合和版本管理。
"""

import json
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 提示模板服务器（简化版）

提供提示模板的创建、管理、发现和使用功能。
支持参数化提示、模板组合和版本管理。
"""

import json
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

class PromptTemplate:
    """提示模板类"""
    
    def __init__(self, name: str, content: str, parameters: Dict[str, Any], 
                 description: str = "", tags: Optional[List[str]] = None, 
                 template_type: str = "system", version: str = "1.0.0"):
        self.name = name
        self.content = content
        self.parameters = parameters or {}
        self.description = description
        self.tags = tags or []
        self.template_type = template_type
        self.version = version
        self.created_at = datetime.now()
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.template_type,
            "description": self.description,
            "content": self.content,
            "parameters": self.parameters,
            "tags": self.tags,
            "version": self.version,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat()
        }

class PromptTemplateServer:
    """MCP提示模板服务器"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.initialize_default_templates()
    
    def initialize_default_templates(self):
        """初始化默认提示模板"""
        default_templates = [
            PromptTemplate(
                name="code_reviewer",
                content="""你是一位经验丰富的代码审查员。请审查给定的代码，重点关注：

1. 代码质量和可读性
2. 潜在的bug和安全问题
3. 性能优化建议
4. 最佳实践遵循情况

代码语言：{language}
审查重点：{focus_areas}

请提供详细的审查报告。""",
                parameters={
                    "language": {"type": "string", "required": True, "description": "编程语言"},
                    "focus_areas": {"type": "string", "required": False, "default": "所有方面", "description": "审查重点领域"}
                },
                description="用于代码审查的系统提示模板",
                tags=["code", "review", "quality"],
                template_type="system"
            ),
            PromptTemplate(
                name="data_analyst",
                content="""你是一位专业的数据分析师。请分析提供的数据并生成报告：

分析类型：{analysis_type}
数据来源：{data_source}
关注指标：{key_metrics}

请提供以下内容：
1. 数据概览和质量评估
2. 关键发现和趋势分析
3. 异常值检测和解释
4. 可行的建议和下一步行动

数据格式：{data_format}""",
                parameters={
                    "analysis_type": {"type": "string", "required": True, "description": "分析类型"},
                    "data_source": {"type": "string", "required": True, "description": "数据来源"},
                    "key_metrics": {"type": "string", "required": False, "default": "所有指标", "description": "关注的关键指标"},
                    "data_format": {"type": "string", "required": False, "default": "JSON", "description": "数据格式"}
                },
                description="数据分析专用提示模板",
                tags=["data", "analysis", "report"],
                template_type="system"
            ),
            PromptTemplate(
                name="creative_writer",
                content="""你是一位富有创意的作家。请根据以下要求创作内容：

写作类型：{content_type}
目标受众：{target_audience}
主题/关键词：{topic}
风格要求：{style}
长度要求：{length}

创作要求：
- 内容要原创且引人入胜
- 符合目标受众的阅读习惯
- 保持指定的写作风格
- 包含相关的主题元素

请开始创作：""",
                parameters={
                    "content_type": {"type": "string", "required": True, "description": "内容类型"},
                    "target_audience": {"type": "string", "required": True, "description": "目标受众"},
                    "topic": {"type": "string", "required": True, "description": "主题或关键词"},
                    "style": {"type": "string", "required": False, "default": "自然流畅", "description": "写作风格"},
                    "length": {"type": "string", "required": False, "default": "适中", "description": "内容长度要求"}
                },
                description="创意写作助手模板",
                tags=["writing", "creative", "content"],
                template_type="system"
            ),
            PromptTemplate(
                name="problem_solver",
                content="""请帮我解决以下问题：

问题描述：{problem_description}
问题类型：{problem_type}
约束条件：{constraints}
期望结果：{expected_outcome}

请提供：
1. 问题分析
2. 解决方案（多个选项）
3. 实施步骤
4. 风险评估
5. 预期效果""",
                parameters={
                    "problem_description": {"type": "string", "required": True, "description": "详细的问题描述"},
                    "problem_type": {"type": "string", "required": True, "description": "问题类型"},
                    "constraints": {"type": "string", "required": False, "default": "无特殊限制", "description": "约束条件"},
                    "expected_outcome": {"type": "string", "required": False, "default": "最优解决方案", "description": "期望结果"}
                },
                description="通用问题解决模板",
                tags=["problem", "solving", "analysis"],
                template_type="user"
            ),
            PromptTemplate(
                name="meeting_facilitator",
                content="""你是一位专业的会议主持人。请协助进行以下类型的会议：

会议类型：{meeting_type}
参与人数：{participant_count}
会议时长：{duration}
主要议题：{main_topics}
期望产出：{expected_outputs}

你的职责：
1. 引导讨论保持在议题范围内
2. 确保所有参与者都有发言机会
3. 总结关键观点和决策
4. 推动会议向期望结果迈进
5. 管理会议时间和节奏

现在开始主持会议。""",
                parameters={
                    "meeting_type": {"type": "string", "required": True, "description": "会议类型"},
                    "participant_count": {"type": "integer", "required": True, "description": "参与人数"},
                    "duration": {"type": "string", "required": True, "description": "会议时长"},
                    "main_topics": {"type": "string", "required": True, "description": "主要议题"},
                    "expected_outputs": {"type": "string", "required": True, "description": "期望产出"}
                },
                description="会议主持和引导模板",
                tags=["meeting", "facilitation", "discussion"],
                template_type="system"
            ),
            PromptTemplate(
                name="tutor_assistant",
                content="""你是一位耐心的教学助手，专门帮助学生学习 {subject}。

教学方式：{teaching_style}
学生水平：{student_level}
学习目标：{learning_objectives}

教学原则：
- 根据学生水平调整解释深度
- 使用具体例子和类比
- 鼓励学生思考和提问
- 提供循序渐进的指导

当前学习主题：{current_topic}

请开始教学指导。""",
                parameters={
                    "subject": {"type": "string", "required": True, "description": "学科名称"},
                    "teaching_style": {"type": "string", "required": False, "default": "互动式", "description": "教学风格"},
                    "student_level": {"type": "string", "required": True, "description": "学生水平"},
                    "learning_objectives": {"type": "string", "required": True, "description": "学习目标"},
                    "current_topic": {"type": "string", "required": True, "description": "当前学习主题"}
                },
                description="个性化教学助手模板",
                tags=["education", "teaching", "tutor"],
                template_type="composite"
            )
        ]
        
        for template in default_templates:
            self.templates[template.name] = template
    
    def list_templates(self, filter_tags: Optional[List[str]] = None, template_type: Optional[str] = None) -> Dict[str, Any]:
        """列出所有提示模板"""
        filtered_templates = []
        
        for template in self.templates.values():
            # 按标签过滤
            if filter_tags and not any(tag in template.tags for tag in filter_tags):
                continue
            
            # 按类型过滤
            if template_type and template.template_type != template_type:
                continue
            
            filtered_templates.append({
                "name": template.name,
                "type": template.template_type,
                "description": template.description,
                "tags": template.tags,
                "parameters": len(template.parameters),
                "usage_count": template.usage_count,
                "version": template.version
            })
        
        return {
            "success": True,
            "total": len(filtered_templates),
            "templates": filtered_templates
        }
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """获取特定提示模板的详细信息"""
        if name not in self.templates:
            return {
                "success": False,
                "error": f"未找到提示模板: {name}"
            }
        
        template = self.templates[name]
        return {
            "success": True,
            "template": template.to_dict()
        }
    
    def render_template(self, name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用参数渲染提示模板"""
        if name not in self.templates:
            return {
                "success": False,
                "error": f"未找到提示模板: {name}"
            }
        
        template = self.templates[name]
        parameters = parameters or {}
        
        try:
            # 检查必需参数
            for param_name, param_info in template.parameters.items():
                if param_info.get("required", False) and param_name not in parameters:
                    return {
                        "success": False,
                        "error": f"缺少必需参数: {param_name}"
                    }
            
            # 添加默认值
            final_params = {}
            for param_name, param_info in template.parameters.items():
                if param_name in parameters:
                    final_params[param_name] = parameters[param_name]
                elif "default" in param_info:
                    final_params[param_name] = param_info["default"]
            
            # 渲染模板
            rendered_content = template.content.format(**final_params)
            
            # 更新使用次数
            template.usage_count += 1
            
            return {
                "success": True,
                "template_name": name,
                "rendered_content": rendered_content,
                "parameters_used": final_params
            }
            
        except KeyError as e:
            return {
                "success": False,
                "error": f"模板参数错误: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"模板渲染失败: {e}"
            }
    
    def create_template(self, name: str, content: str, parameters: Optional[Dict[str, Any]] = None, 
                       description: str = "", tags: Optional[List[str]] = None, 
                       template_type: str = "user") -> Dict[str, Any]:
        """创建新的提示模板"""
        if name in self.templates:
            return {
                "success": False,
                "error": f"提示模板已存在: {name}"
            }
        
        template = PromptTemplate(
            name=name,
            content=content,
            parameters=parameters or {},
            description=description,
            tags=tags or [],
            template_type=template_type
        )
        
        self.templates[name] = template
        
        return {
            "success": True,
            "message": f"成功创建提示模板: {name}",
            "template": template.to_dict()
        }
    
    def update_template(self, name: str, **kwargs) -> Dict[str, Any]:
        """更新现有的提示模板"""
        if name not in self.templates:
            return {
                "success": False,
                "error": f"未找到提示模板: {name}"
            }
        
        template = self.templates[name]
        
        # 更新字段
        if "content" in kwargs:
            template.content = kwargs["content"]
        if "description" in kwargs:
            template.description = kwargs["description"]
        if "parameters" in kwargs:
            template.parameters = kwargs["parameters"]
        if "tags" in kwargs:
            template.tags = kwargs["tags"]
        
        return {
            "success": True,
            "message": f"成功更新提示模板: {name}",
            "template": template.to_dict()
        }
    
    def delete_template(self, name: str) -> Dict[str, Any]:
        """删除提示模板"""
        if name not in self.templates:
            return {
                "success": False,
                "error": f"未找到提示模板: {name}"
            }
        
        del self.templates[name]
        
        return {
            "success": True,
            "message": f"成功删除提示模板: {name}"
        }
    
    def get_template_stats(self) -> Dict[str, Any]:
        """获取提示模板使用统计"""
        total_templates = len(self.templates)
        total_usage = sum(template.usage_count for template in self.templates.values())
        
        # 按类型统计
        type_stats = {}
        for template in self.templates.values():
            template_type = template.template_type
            if template_type not in type_stats:
                type_stats[template_type] = {"count": 0, "usage": 0}
            type_stats[template_type]["count"] += 1
            type_stats[template_type]["usage"] += template.usage_count
        
        # 最常用的模板
        most_used = sorted(self.templates.values(), key=lambda t: t.usage_count, reverse=True)[:5]
        most_used_list = [
            {
                "name": template.name,
                "usage_count": template.usage_count,
                "type": template.template_type
            }
            for template in most_used
        ]
        
        return {
            "success": True,
            "total_templates": total_templates,
            "total_usage": total_usage,
            "average_usage": total_usage / max(total_templates, 1),
            "type_statistics": type_stats,
            "most_used_templates": most_used_list
        }

# 全局服务器实例
_server_instance = None

def get_server() -> PromptTemplateServer:
    """获取服务器实例"""
    global _server_instance
    if _server_instance is None:
        _server_instance = PromptTemplateServer()
    return _server_instance

if __name__ == "__main__":
    # 测试服务器功能
    server = get_server()
    
    # 测试列出模板
    result = server.list_templates()
    print("模板列表:", json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试渲染模板
    result = server.render_template("code_reviewer", {
        "language": "Python",
        "focus_areas": "性能优化"
    })
    print("\n渲染结果:", json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n提示模板服务器测试完成！")
