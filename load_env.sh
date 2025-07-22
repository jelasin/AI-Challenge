#!/bin/bash

# 加载.env文件中的环境变量
ENV_FILE="${1:-.env}"

# 检查.env文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    echo "警告: 找不到 $ENV_FILE 文件" >&2
    exit 1
fi

echo "正在从 $ENV_FILE 加载环境变量..."

# 读取.env文件并设置环境变量
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和注释行
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # 检查是否包含等号
    if [[ "$line" =~ ^[^=]+=.*$ ]]; then
        # 分割键值对
        key=$(echo "$line" | cut -d'=' -f1 | xargs)
        value=$(echo "$line" | cut -d'=' -f2- | xargs)
        
        # 移除值两端的引号
        if [[ "$value" =~ ^\".*\"$ ]] || [[ "$value" =~ ^\'.*\'$ ]]; then
            value=${value#[\"\']}
            value=${value%[\"\']}
        fi
        
        # 导出环境变量
        export "$key"="$value"
        echo "已设置: $key"
    else
        echo "跳过无效行: $line" >&2
    fi
done < "$ENV_FILE"

echo "环境变量加载完成!"
