"""
MCP资源管理示例代码
"""

def analyze_resource(content):
    """分析资源内容"""
    return {
        "length": len(content),
        "lines": content.count("\n"),
        "words": len(content.split())
    }

class ResourceAnalyzer:
    def __init__(self):
        self.processed_count = 0
    
    def process(self, resource):
        self.processed_count += 1
        return analyze_resource(resource)
