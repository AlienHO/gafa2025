#!/bin/bash

# 获取脚本所在目录作为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

echo "使用conda路径: $CONDA_PATH"
ENV_NAME="py310"

# 激活conda环境并运行程序
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 如果环境不存在，则从environment.yml创建
if [ $? -ne 0 ]; then
    echo "环境 $ENV_NAME 不存在，正在从environment.yml创建..."
    conda env create -f environment.yml
    conda activate $ENV_NAME
fi

# 运行主程序
python main.py

# 退出环境
conda deactivate
