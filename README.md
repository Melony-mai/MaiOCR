# MaiOCR

离线 OCR 桌面工具（Windows）。截图 → 本地识别 → 自动复制到剪贴板，全程不上传任何数据。

## 功能

- **文字识别**：中文 / 日文 / 英文及中英混排
- **代码识别**：代码模式下始终输出为 Markdown 代码块，并包含：
  - 缩进重建：利用 OCR 检测框的坐标还原每行缩进层级与空行（纯文本模式不启用）
  - 语言/格式检测（Python / JS / TS / Java / C# / C/C++ / Go / Rust / PHP / Ruby / Swift / Kotlin / SQL / HTML / CSS / JSON / YAML / Bash / PowerShell 等）
  - 截图痕迹清理：自动剥离编辑器行号栏、Shell / REPL / PowerShell 提示符
  - 符号修复：全角字符、智能引号、`→ ≥ ≠ …` 等 Unicode 运算符还原为 ASCII
  - 结构识别：统计函数 / 类 / 导入数量，检测括号是否闭合（结果窗口状态栏展示）
  - 仅当内容明显为自然语言时才回退为纯文本
- **统一框选截图**：触发后屏幕变暗，拖拽框选任意区域（支持多显示器），Esc / 右键取消；热键与托盘菜单走同一选区界面
- **GPU 加速**：DirectML（任何 DX12 GPU）/ CUDA，不可用时自动回退 CPU；启动时预热并对异常适配器自动切换
- **日志**：记录启动、截图耗时、OCR 耗时、异常等；单文件超 10 MB 自动轮转，7 天前的旧日志自动清理
- **长期后台运行**：
  - 热键监听线程异常退出后自动重启（指数退避上限 5 次），仅在彻底失败时提示用户
  - GPU 推理故障（睡眠唤醒 / 驱动异常）自动重建会话，连续失败才降级 CPU
  - 单实例守卫 + 进程优先级调优（禁用 EcoQoS 节流）
- **结果窗口**：展示识别历史（本次会话内），一键复制
- **系统托盘**：常驻后台，单实例运行

## 快捷键

| 快捷键 | 功能 |
|---|---|
| `Win + PrtSc` | 拖拽框选区域 → 文字识别 |
| `Win + Ctrl + PrtSc` | 拖拽框选区域 → 代码识别 |
| 托盘菜单「截图识别文字 / 截图识别代码」 | 与对应热键完全相同（框选 → 识别） |
| 托盘菜单 / 双击托盘图标 | 查看历史记录 |

> Windows 10/11 系统自身会占用 `Win+PrtSc`，此时 MaiOCR 自动改用备选组合
> （文字：`Win+Shift+PrtSc`；代码：`Win+Ctrl+Shift+PrtSc`），并通过托盘气泡提示。
> 每组快捷键独立注册，任一组合被占用不影响其余快捷键。

## 技术栈

PySide6（UI/托盘/选区）、RapidOCR + ONNX Runtime（推理）、mss（截屏）、loguru（日志）、pyperclip（剪贴板）。模型随 rapidocr 包内置（PP-OCRv6），首次运行无需联网。

## 开发运行（Linux 亦可调试 UI/OCR，热键仅 Windows）

```bash
uv sync --extra linux   # Linux 开发环境（含 evdev 热键）
uv sync                 # Windows
uv run python main.py   # Shift+Enter 触发（Linux）/ 托盘菜单
```

Linux 下热键为 `Shift+Enter`（evdev 实现，需输入设备权限），仅供开发验证流程。

## Windows 打包 EXE

在 Windows 设备上：

```powershell
# 1. 安装 Python 3.12+ 与 uv（或直接用 pip）
pip install uv

# 2. 同步依赖（含 PyInstaller）
uv sync --extra build

# 3. 构建产物位于 dist/MaiOCR/MaiOCR.exe（onedir，整目录可拷贝分发）
uv run pyinstaller maiocr.spec --noconfirm
```

### GPU 加速（可选）

默认 `onnxruntime` 仅含 CPU。启用 CUDA：

```powershell
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu        # 需与本机 CUDA/cuDNN 版本匹配
uv run pyinstaller maiocr.spec --noconfirm
```

要求 NVIDIA GPU + CUDA 12.x + cuDNN 9.x。程序启动时会检测：

- 检测到 CUDA → 用 GPU 初始化引擎；
- 初始化或推理失败（缺 DLL、驱动不匹配等）→ 自动重建 CPU 引擎并记录日志；
- 日志中会打印实际生效的 providers（`CUDAExecutionProvider` / `CPUExecutionProvider`）。

### 日文识别（可选）

内置 ch/en 模型对日文假名效果有限。程序会在识别结果假名占比过高时尝试用日文模型二次识别——前提是本地存在模型文件（程序强制离线，任何缺失文件都绝不联网下载）：

1. 从 ModelScope 下载 `japan_PP-OCRv4_rec_mobile.onnx`
   （RapidOCR v3.9.2 对应版本）；
2. 放到 `MaiOCR.exe` 同级的 `models\` 目录下（或 `%LOCALAPPDATA%\MaiOCR\models\`）。

> 若该 ONNX 未内嵌字典（极少见），再在同目录放一个同名 `.txt`
> 字典文件（如 `japan_PP-OCRv4_rec_mobile.txt`）即可；两者都不存在时
> 日文二次识别自动跳过并保留默认结果。

## 数据与日志位置（Windows）

| 内容 | 路径 |
|---|---|
| 日志 | `%LOCALAPPDATA%\MaiOCR\logs\maiocr.log`（10MB 轮转，保留 7 天） |
| 可选外部模型 | `<exe目录>\models\` 或 `%LOCALAPPDATA%\MaiOCR\models\` |

## 常见问题

- **提示"注册全局热键失败"**：其他软件占用了 PrtSc 组合键，退出占用方或修改其配置。
- **识别为空**：截图过小/对比度过低；区域选择尽量贴近文本。
- **双击 exe 出现两个实例？** 不会——通过 QLocalServer 强制单实例，重复启动会自动退出。
