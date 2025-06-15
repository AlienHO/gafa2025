#!/bin/bash

# 设置conda环境名称
ENV_NAME="py310"

# 自动检测conda安装路径
find_conda() {
    # 使用which命令尝试查找conda
    if command -v conda &> /dev/null; then
        conda_exec=$(command -v conda)
        # 如果是软链接，获取真实路径
        if [ -L "$conda_exec" ]; then
            conda_exec=$(readlink -f "$conda_exec")
        fi
        # 返回conda安装路径
        echo $(dirname $(dirname "$conda_exec"))
        return 0
    fi
    
    # 如果which查找失败，检查常见安装路径
    local common_paths=(
        "/opt/homebrew/anaconda3"
        "$HOME/anaconda3"
        "/opt/anaconda3"
        "$HOME/miniconda3"
        "/opt/miniconda3"
        "/opt/homebrew/Caskroom/miniconda/base"
    )
    
    for path in "${common_paths[@]}"; do
        if [ -f "$path/bin/conda" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # 如果找不到conda，接下来会报错
    echo ""
    return 1
}

# 获取conda路径
CONDA_PATH=$(find_conda)
if [ -z "$CONDA_PATH" ]; then
    echo "错误: 无法找到conda安装。请安装Anaconda或Miniconda。"
    exit 1
fi

echo "使用conda安装路径: $CONDA_PATH"

# 激活conda环境
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "错误: 环境 $ENV_NAME 不存在，请先运行 ./run_gafa.sh 创建环境"
    exit 1
fi

echo "正在导出环境 $ENV_NAME 到 environment-explicit.yml..."

# 导出完整的环境规范（包括精确的版本号和构建字符串）
conda env export --no-builds > environment-exact.yml
echo "已导出详细环境到 environment-exact.yml"

# 导出平台无关的环境规范（仅包含pip安装的包）
conda env export --from-history > environment-simple.yml
echo "已导出简化环境到 environment-simple.yml"

# 创建requirements.txt用于pip
pip freeze > requirements.txt
echo "已导出pip包列表到 requirements.txt"

echo "环境导出完成！其他开发者可以使用以下命令重建环境:"
echo "conda env create -f environment-exact.yml  # 精确复制当前环境"
echo "或"
echo "conda env create -f environment-simple.yml  # 创建最小环境"
echo "或"
echo "pip install -r requirements.txt  # 仅安装Python包"

echo "导出完成。"
