# GAF字母检测与追踪系统

基于YOLOv8的实时G、A、F字母检测与追踪系统，集成OpenAI Vision功能和OSC通信。
## 安装与运行指南

### 1. 安装conda环境
#### Windows

1. 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/distribution)
2. 安装好Miniconda或Anaconda后，找到安装目录
   - 默认路径通常为 `C:\Users\<用户名>\Miniconda3` 或 `C:\Users\<用户名>\Anaconda3`
3. 运行安装程序，勾选“Add to PATH”选项
4. 下载[Gitbash](https://git-scm.com/downloads)
5. 在vscode中打开本项目的主文件夹，创建新的名为bashrc的文件:
```bash
code ~/.bashrc
```
6. 在bashrc文件中添加以下内容（二选一）：

- 如果安装的是miniconda，添加以下内容：
```bash
# >>> miniconda manual setup for Git Bash >>>
export PATH="/c/Users/<用户名>/miniconda3/bin:/c/Users/<用户名>/miniconda3/Scripts:/c/Users/<用户名>/miniconda3/Library/bin:$PATH"
# <<< miniconda manual setup <<<
```
- 如果安装的是anaconda，添加以下内容：
```bash
# >>> Anaconda manual setup for Git Bash >>>
export PATH="/c/Users/<用户名>/Anaconda3/bin:/c/Users/<用户名>/Anaconda3/Scripts:/c/Users/<用户名>/Anaconda3/Library/bin:$PATH"
# <<< Anaconda manual setup <<<
```
注意：将 `<用户名>` 替换为你的Windows用户名。

7. 重新载入bashrc环境
```bash
source ~/.bashrc
```
8. 验证conda是否安装成功
```bash
conda --version
```

#### macOS

```bash
# 使用Homebrew安装Anaconda
brew install --cask anaconda
```
#### Linux

```bash
# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# 执行安装
bash ~/miniconda.sh -b -p $HOME/miniconda

# 添加到PATH
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 运行项目

项目提供了便捷的`run_gafa.sh`脚本，它能自动检测conda安装、创建所需环境并运行程序：

```bash
# 添加执行权限（Linux、macOS需要执行，Windows可以直接运行脚本）
chmod +x run_gafa.sh

# 运行脚本
./run_gafa.sh
```

**脚本功能：**

- 自动检测conda安装路径
- 创建`py310` Python环境（如果不存在）
- 安装所需依赖
- 启动GAF字母检测系统

### 手动环境设置（可选）

如果需要手动管理环境，可以使用以下命令：

```bash
# 创建conda环境
conda env create -f environment.yml

# 或者手动创建环境
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt

# 启动程序
python main.py
```

### 环境导出与共享

为方便其他开发者复制完全相同的环境，项目提供了环境导出脚本：

```bash
# 添加执行权限
chmod +x export_env.sh

# 导出环境
./export_env.sh
```

该脚本会生成三个文件：
- `environment-exact.yml`: 完整环境配置，包含所有依赖包的精确版本
- `environment-simple.yml`: 简化环境配置，只包含直接安装的包
- `requirements.txt`: 标准pip格式的依赖列表

其他开发者可以使用以下命令重建环境：

```bash
# 方法1: 使用精确环境复制（推荐）
conda env create -f environment-exact.yml

# 方法2: 使用简化环境
conda env create -f environment-simple.yml
```

### 方法2: 使用pip安装依赖

```bash
pip install -r requirements.txt
```

## 功能特点

- **实时字母检测**: 基于YOLOv8模型检测G、A、F字母
- **目标追踪**: IoU based追踪器，支持ID持久化和稳定性判断
- **人物检测**: 检测画面中的人物
- **OpenAI Vision集成**: 大模型随机生成画面区域描述
- **OSC通信**: 通过OSC协议发送检测结果到TouchDesigner等软件
- **模块化架构**: 清晰的代码结构，易于维护和扩展

## 项目结构

```
gafa-1/
├─ main.py                    # 主程序入口
├─ modules/                   # 模块化代码
├─ GAFA.pt                   # GAF字母检测模型（需要提供）
├─ run_gafa.sh               # 一键运行脚本（自动设置环境并运行）
├─ export_env.sh             # 导出环境配置脚本
├─ environment.yml          # conda环境配置
└─ requirements.txt          # 依赖包列表
```

## 使用方法

### 1. 配置参数

运行前编辑`modules/config.py`文件中的参数（仅列出关键项）：

```python
# OpenAI API设置
OPENAI_API_KEY = "your-api-key"  # 替换为您的API密钥

# OSC通信设置
OSC_IP = "127.0.0.1"  # 接收OSC消息的设备IP
OSC_PORT = 8000        # 接收端口
LETTER_OSC_ADDRESS = "/letters"  # 字母检测OSC地址
PERSON_OSC_ADDRESS = "/persons"  # 人物检测OSC地址
```

### 2. 必要的模型文件

- **GAFA.pt**: GAF字母检测模型（请将此文件放到项目根目录）
- **yolov8n.pt**: 人物检测模型（首次运行时自动下载）

### 3. 运行系统

```bash
# 直接使用一键运行脚本
./run_gafa.sh
```

### 4. 控制程序

- 按键盘`q`键优雅退出
- 或使用`Ctrl+C`强制退出

### 5. 可能遇到的问题及解决方案

1. **找不到conda命令**:
   - 确保conda已正确安装
   - 确认conda已添加到PATH环境变量
   - 尝试重启终端或打开新的终端窗口

2. **conda环境创建失败**:
   - 检查网络连接
   - 确俞environment.yml文件存在且格式正确

3. **找不到摄像头**:
   - 检查摄像头连接
   - 确认是否有其他应用程序正在使用摄像头

4. **缺少GAFA.pt模型**:
```
# 字母检测结果
/letters [class_id, x, y, width, height]  # 只输出稳定的目标
# 注意: x,y为中心点坐标，Y坐标已反转与人物检测一致

# 人物检测结果
/persons [id, x, y, width, height, confidence]

# Vision API结果 (区域描述)
/anything [x, y, width, height, description]
```

注意：坐标和尺寸均为标准化值（0-1范围），可乘以实际尺寸得到像素值。

### 6. 常见问题排除

| 问题 | 解决方法 |
|--------|----------|
| 无法找到conda命令 | 检查`.zshrc`中conda路径是否正确，运行`source ~/.zshrc` |
| 环境创建失败 | 手动删除已有环境`conda remove -n py310 --all` |
| 摄像头无法打开 | 修改`config.py`中的`CAMERA_ID`参数或检查摄像头权限 |
| GAFA.pt模型缺失 | 确认模型文件已放在项目根目录下 |
| OpenAI API错误 | 检查API密钥和网络连接、确认配额充足 |
| OSC消息不可见 | 确认目标应用（如TouchDesigner）正在该IP和端口监听 |

## 开发和扩展

系统采用模块化设计，便于扩展：

- 添加新的检测类别：修改`modules/config.py`中的类别定义
- 集成新的API：在`modules/vision_api/`中添加新的API封装
- 添加新的输出格式：在`modules/utils/osc_utils.py`中添加新的OSC函数

## 许可证

本项目遵循MIT许可证。

## 作者

Ho Alien ，Leonardo Li - 版本 2.0.0 - 2025-06-14
