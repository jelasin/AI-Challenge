# 加载.env文件中的环境变量
param(
    [string]$EnvFile = ".env"
)

# 检查.env文件是否存在
if (-not (Test-Path $EnvFile)) {
    Write-Host "警告: 找不到 $EnvFile 文件" -ForegroundColor Yellow
    exit 1
}

Write-Host "正在从 $EnvFile 加载环境变量..." -ForegroundColor Green

try {
    # 读取.env文件内容
    $envContent = Get-Content $EnvFile -ErrorAction Stop
    
    foreach ($line in $envContent) {
        # 跳过空行和注释行
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            continue
        }
        
        # 解析键值对
        if ($line -match "^([^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # 移除值两端的引号
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or 
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            # 设置环境变量
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "已设置: $key" -ForegroundColor Cyan
        }
        else {
            Write-Host "跳过无效行: $line" -ForegroundColor Yellow
        }
    }
    
    Write-Host "环境变量加载完成!" -ForegroundColor Green
}
catch {
    Write-Host "错误: 加载环境变量时出现异常: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
