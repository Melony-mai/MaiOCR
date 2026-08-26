# MaiOCR

**Offline screenshot OCR desktop tool for Windows.** Select a screen region → recognize locally → results are copied to your clipboard automatically. No data ever leaves your machine.

**离线 OCR 桌面工具（Windows）**。截图 → 本地识别 → 自动复制到剪贴板，全程不上传任何数据。

<p align="center">
  <a href="#english">English</a> ·
  <a href="#简体中文">简体中文</a>
</p>

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

<a id="english"></a>
## English

### ✨ Features

#### Core Recognition
- **Text recognition**: Chinese, Japanese, English, and mixed CJK/Latin text
- **Code recognition**: purpose-built for screenshots of source code; always outputs a Markdown fenced code block
  - **Indentation reconstruction**: restores per-line indent levels and blank lines from OCR bounding-box coordinates
  - **Language detection**: Python / JavaScript / TypeScript / Java / C# / C++ / C / Go / Rust / PHP / Ruby / Swift / Kotlin / SQL / HTML / CSS / JSON / YAML / Bash / PowerShell
  - **Editor artifact cleanup**: strips editor line-number gutters, shell / REPL / IPython prompts, and PowerShell prompts
  - **Symbol repair**: full-width characters, smart quotes, and Unicode operators (`→ ≥ ≠ …`) normalized back to ASCII
  - **Docstring repair**: restores Python `"""` triple quotes mangled by OCR
  - **Structure analysis**: function / class / import counts plus a bracket-balance check, shown in the result window status bar
  - **Smart fallback**: falls back to plain text only when the content is clearly natural-language prose

#### Interaction
- **Unified region selection**: the screen dims on trigger, drag to select any region (multi-monitor aware), Esc or right-click cancels; hotkeys and tray menu share the same selection UI
- **Result window**: in-session recognition history (up to 100 entries); every successful scan is auto-copied to the clipboard; copy / delete selected / clear-all controls (Delete key deletes selected entries)
- **System tray**: resident background app with single-instance enforcement; double-click opens history
- **Auto-start on boot**: toggle from the tray menu (writes to the current user's registry Run key — no admin rights required)
- **In-app restart**: the "Restart MaiOCR" tray item exits and relaunches cleanly — no need to close and reopen manually when something goes wrong
- **Hotkey fallback**: if Windows itself occupies `Win+PrtSc`, MaiOCR automatically switches to alternate combos and notifies you via balloon

#### Performance & Reliability
- **GPU acceleration**: DirectML by default (any DX12 GPU — NVIDIA / AMD / Intel), optional CUDA build, automatic CPU fallback; the OCR engine loads lazily on first inference so startup stays light
- **VRAM management**: idle GPU sessions auto-release after 5 minutes; manual release is available from the tray menu with live VRAM status
- **Logging**: captures startup events, screenshot/OCR timings, and exceptions; rotates at 10 MB per file and prunes logs older than 7 days
- **Long-run stability**:
  - Self-healing hotkey listener: automatically restarts after crashes (up to 5 attempts) and only notifies you when all retries fail
  - GPU session rebuild after sleep/wake or driver failures; degrades to CPU only after repeated consecutive failures
  - Single-instance guard + process priority tuning (EcoQoS power throttling disabled)

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Win + PrtSc` | Drag-select a region → text recognition |
| `Win + Ctrl + PrtSc` | Drag-select a region → code recognition |
| Tray → Capture Text | Same as `Win+PrtSc` |
| Tray → Capture Code | Same as `Win+Ctrl+PrtSc` |
| Tray → History / double-click tray icon | Open the history window |
| Tray → Launch at Startup | Toggle autostart on login |
| Tray → Restart MaiOCR | Exit and relaunch the app |
| Result window: `Esc` | Hide window |
| Result window: `Delete` | Delete selected history entries |

> **Note**: Windows 10/11 reserves `Win+PrtSc` for its own screenshot feature. When that happens MaiOCR switches to fallback combos automatically (text: `Win+Shift+PrtSc`; code: `Win+Ctrl+Shift+PrtSc`) and shows a tray-balloon notice. Each shortcut pair registers independently — one combo being occupied never affects the others.

### 🛠 Tech Stack

- **UI / tray / selection overlay**: PySide6 (Qt 6)
- **OCR inference**: RapidOCR + ONNX Runtime
- **Screen capture**: mss
- **Logging**: loguru
- **Clipboard**: pyperclip (with Qt clipboard preferred natively)
- **Image processing**: OpenCV (cv2), Pillow
- **Models**: PP-OCRv6 detection + recognition (bundled inside the `rapidocr` package — no network needed on first run)

### 🚀 Getting Started

**Prerequisites**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
# Clone the repository
git clone https://github.com/Melony-mai/MaiOCR.git
cd MaiOCR

# Windows
uv sync

# Linux (development only — UI/OCR pipeline works, hotkeys are Windows-only;
# uses Shift+Enter via evdev, requires input-device permissions)
uv sync --extra linux

# Run
uv run python main.py
```

Run the test suite:

```bash
uv run pytest
```

Diagnostic benchmark mode (3 timed full-screen recognition passes):

```bash
uv run python main.py --bench
```

### 📦 Building the Windows EXE

Build a distributable onedir application on a Windows machine:

```powershell
# 1. Install Python 3.12+ and uv
pip install uv

# 2. Sync dependencies (includes PyInstaller)
uv sync --extra build

# 3. Build
uv run pyinstaller maiocr.spec --noconfirm

# Output: dist\MaiOCR\MaiOCR.exe (onedir mode — distribute the whole folder)
```

#### Optional: Enable CUDA GPU Acceleration

DirectML (`onnxruntime-directml`) is the default and runs on any DX12 GPU. To build with CUDA instead:

```powershell
uv pip uninstall onnxruntime-directml
uv pip install onnxruntime-gpu        # must match your local CUDA/cuDNN versions
uv run pyinstaller maiocr.spec --noconfirm
```

**Requirements**: NVIDIA GPU + CUDA 12.x + cuDNN 9.x. The app detects the backend at runtime:
- CUDA available → engine initializes on GPU
- Initialization or inference failure (missing DLLs, driver mismatch, …) → automatically rebuilds on CPU and logs the incident
- The active providers are logged at startup (`CUDAExecutionProvider` / `DmlExecutionProvider` / `CPUExecutionProvider`)

### 📁 Data & Log Locations (Windows)

| Content | Path |
|---|---|
| **Log files** | `%LOCALAPPDATA%\MaiOCR\logs\maiocr.log` (10 MB rotation, 7-day retention) |
| **Optional external models** | `<exe folder>\models\` or `%LOCALAPPDATA%\MaiOCR\models\` |

Recognition history lives in memory only and is never written to disk; it clears when the app exits.

### 🌐 Japanese Recognition (Optional Enhancement)

The bundled Chinese/English models handle Japanese kana poorly. When kana dominate a result, MaiOCR re-runs recognition with a Japanese model — **but only if the model file exists locally** (the app is strictly offline and will never download anything):

1. Download `japan_PP-OCRv4_rec_mobile.onnx` from ModelScope (the version matching RapidOCR v3.9.2)
2. Place it next to `MaiOCR.exe` in a `models\` subfolder (or in `%LOCALAPPDATA%\MaiOCR\models\`)

> If the ONNX lacks an embedded dictionary (rare), also place a same-named `.txt` dictionary file (e.g. `japan_PP-OCRv4_rec_mobile.txt`) in the same folder. If neither file exists, Japanese re-recognition is skipped silently and default results are kept.

### 🧩 Project Layout

```
src/maiocr/
├── app.py        # Application entry point & wiring
├── core/         # Capture → OCR → post-process pipeline
├── hotkey/       # Global hotkey registration (win32; linux = dev only)
├── ocr/          # RapidOCR engine wrapper + code-mode post-processing
├── ui/           # Tray icon, region selector, result window
└── utils/        # Paths, logging, autostart, single-instance, performance tuning
```

### ❓ FAQ

| Problem | Solution |
|---|---|
| **"Failed to register global hotkey"** | Another program occupies the PrtSc combo — quit it or change its settings |
| **Empty recognition result** | Region too small or low contrast; select an area tight around the text |
| **Two instances after double-clicking the exe?** | Never happens — QLocalServer enforces a single instance; duplicate launches exit automatically |
| **GPU acceleration not active** | Check the providers line in the log; CUDA needs matching toolkit versions, DirectML needs a DX12-capable driver |
| **Japanese recognition not working** | Verify `models\japan_PP-OCRv4_rec_mobile.onnx` exists and matches the expected version |

---

<a id="简体中文"></a>
## 简体中文

### ✨ 功能特性

#### 核心识别
- **文字识别**：支持中文、日文、英文及中英混排
- **代码识别**：专为代码截图优化，始终输出 Markdown 围栏代码块
  - 缩进重建：利用 OCR 检测框坐标还原每行缩进层级与空行
  - 语言检测：自动识别 Python / JavaScript / TypeScript / Java / C# / C++ / C / Go / Rust / PHP / Ruby / Swift / Kotlin / SQL / HTML / CSS / JSON / YAML / Bash / PowerShell 等 20 种语言
  - 截图痕迹清理：自动剥离编辑器行号栏、Shell / REPL / IPython / PowerShell 提示符
  - 符号修复：全角字符、智能引号、`→ ≥ ≠ …` 等 Unicode 运算符还原为 ASCII
  - 文档字符串修复：还原被 OCR 损坏的 Python `"""` 三引号
  - 结构识别：统计函数 / 类 / 导入数量，检测括号是否闭合（结果窗口状态栏展示）
  - 智能回退：仅当内容明显为自然语言时才回退为纯文本

#### 交互体验
- **统一框选截图**：触发后屏幕变暗，拖拽框选任意区域（支持多显示器），Esc / 右键取消；热键与托盘菜单走同一选区界面
- **结果窗口**：展示本次会话内的识别历史（最多 100 条）；每次成功识别自动复制到剪贴板；支持复制 / 删除选中 / 清空历史（Delete 键删除选中条目）
- **系统托盘**：常驻后台，强制单实例运行，双击查看历史
- **开机自启**：托盘菜单一键开关（写入当前用户注册表 Run 项，无需管理员权限）
- **应用内重启**：托盘菜单「重启 MaiOCR」自动退出并拉起新进程，异常时无需手动关闭再打开
- **热键自动避让**：Windows 占用 `Win+PrtSc` 时自动改用备选组合并气泡提示

#### 性能与可靠性
- **GPU 加速**：默认 DirectML（任何 DX12 显卡——NVIDIA / AMD / Intel 均可），可选 CUDA 构建，不可用时自动回退 CPU；OCR 引擎在首次识别时才懒加载，启动保持轻量
- **显存管理**：GPU 会话闲置 5 分钟后自动释放显存；也可在托盘菜单手动释放，实时显示显存状态
- **日志系统**：记录启动事件、截图/OCR 耗时与异常；单文件超 10 MB 自动轮转，7 天前的旧日志自动清理
- **长期后台运行**：
  - 热键监听线程崩溃后自愈重启（最多 5 次），仅在彻底失败时提示用户
  - GPU 推理故障（睡眠唤醒 / 驱动异常）自动重建会话，连续失败才降级 CPU
  - 单实例守卫 + 进程优先级调优（禁用 EcoQoS 功耗节流）

### ⌨️ 快捷键

| 快捷键 | 功能 |
|---|---|
| `Win + PrtSc` | 拖拽框选区域 → 文字识别 |
| `Win + Ctrl + PrtSc` | 拖拽框选区域 → 代码识别 |
| 托盘菜单「截图识别文字」 | 与 `Win+PrtSc` 完全相同 |
| 托盘菜单「截图识别代码」 | 与 `Win+Ctrl+PrtSc` 完全相同 |
| 托盘菜单「历史记录」 / 双击托盘图标 | 打开历史记录窗口 |
| 托盘菜单「开机自启动」 | 开关登录后自动运行 |
| 托盘菜单「重启 MaiOCR」 | 自动退出并重新启动应用 |
| 结果窗口 `Esc` | 隐藏窗口 |
| 结果窗口 `Delete` | 删除选中的历史条目 |

> **注意**：Windows 10/11 系统自身会占用 `Win+PrtSc`，此时 MaiOCR 自动改用备选组合（文字：`Win+Shift+PrtSc`；代码：`Win+Ctrl+Shift+PrtSc`），并通过托盘气泡提示。每组快捷键独立注册，任一组合被占用不影响其余快捷键。

### 🛠 技术栈

- **UI / 托盘 / 选区遮罩**：PySide6 (Qt 6)
- **OCR 推理**：RapidOCR + ONNX Runtime
- **截屏**：mss
- **日志**：loguru
- **剪贴板**：pyperclip（优先使用 Qt 原生剪贴板）
- **图像处理**：OpenCV (cv2)、Pillow
- **模型**：PP-OCRv6 检测 + 识别（随 `rapidocr` 包内置，首次运行无需联网）

### 🚀 快速开始

**前置要求**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

```bash
# 克隆仓库
git clone https://github.com/Melony-mai/MaiOCR.git
cd MaiOCR

# Windows
uv sync

# Linux（仅供开发调试——UI/OCR 流程可用，热键仅限 Windows；
# 使用 evdev 监听 Shift+Enter，需要输入设备权限）
uv sync --extra linux

# 运行
uv run python main.py
```

运行测试：

```bash
uv run pytest
```

诊断基准模式（连续 3 次全屏识别计时）：

```bash
uv run python main.py --bench
```

### 📦 构建 Windows EXE

在 Windows 设备上构建可分发的单目录程序：

```powershell
# 1. 安装 Python 3.12+ 与 uv
pip install uv

# 2. 同步依赖（含 PyInstaller）
uv sync --extra build

# 3. 构建
uv run pyinstaller maiocr.spec --noconfirm

# 产物位置：dist\MaiOCR\MaiOCR.exe（onedir 模式，整目录可拷贝分发）
```

#### 可选：启用 CUDA GPU 加速

默认的 `onnxruntime-directml` 可在任何 DX12 显卡上运行。如需改用 CUDA：

```powershell
uv pip uninstall onnxruntime-directml
uv pip install onnxruntime-gpu        # 需与本机 CUDA/cuDNN 版本匹配
uv run pyinstaller maiocr.spec --noconfirm
```

**要求**：NVIDIA GPU + CUDA 12.x + cuDNN 9.x。程序运行时自动检测后端：
- 检测到 CUDA → 引擎以 GPU 初始化
- 初始化或推理失败（缺 DLL、驱动不匹配等）→ 自动重建 CPU 引擎并记录日志
- 日志中会打印实际生效的 providers（`CUDAExecutionProvider` / `DmlExecutionProvider` / `CPUExecutionProvider`）

### 📁 数据与日志位置（Windows）

| 内容 | 路径 |
|---|---|
| **日志文件** | `%LOCALAPPDATA%\MaiOCR\logs\maiocr.log`（10 MB 轮转，保留 7 天） |
| **可选外部模型** | `<exe目录>\models\` 或 `%LOCALAPPDATA%\MaiOCR\models\` |

识别历史仅保存在内存中，绝不写入磁盘，应用退出即清空。

### 🌐 日文识别（可选增强）

内置中英文模型对日文假名效果有限。当识别结果中假名占比过高时，程序会用日文模型二次识别——**前提是本地存在模型文件**（程序强制离线，任何缺失文件都绝不联网下载）：

1. 从 ModelScope 下载 `japan_PP-OCRv4_rec_mobile.onnx`（RapidOCR v3.9.2 对应版本）
2. 放到 `MaiOCR.exe` 同级的 `models\` 目录下（或 `%LOCALAPPDATA%\MaiOCR\models\`）

> 若该 ONNX 未内嵌字典（极少见），再在同目录放一个同名 `.txt` 字典文件（如 `japan_PP-OCRv4_rec_mobile.txt`）即可；两者都不存在时日文二次识别自动跳过并保留默认结果。

### 🧩 项目结构

```
src/maiocr/
├── app.py        # 应用入口与装配
├── core/         # 截图 → OCR → 后处理流水线
├── hotkey/       # 全局热键注册（win32；linux 仅供开发）
├── ocr/          # RapidOCR 引擎封装 + 代码模式后处理
├── ui/           # 托盘图标、区域选择器、结果窗口
└── utils/        # 路径、日志、自启动、单实例、性能调优
```

### ❓ 常见问题

| 问题 | 解决方案 |
|---|---|
| **提示“注册全局热键失败”** | 其他软件占用了 PrtSc 组合键，退出占用方或修改其配置 |
| **识别结果为空** | 区域过小或对比度过低；尽量框选贴近文本的区域 |
| **双击 exe 出现两个实例？** | 不会——通过 QLocalServer 强制单实例，重复启动会自动退出 |
| **GPU 加速不生效** | 检查日志中的 providers 输出；CUDA 需匹配工具链版本，DirectML 需 DX12 显卡驱动 |
| **日文识别不生效** | 确认 `models\japan_PP-OCRv4_rec_mobile.onnx` 存在且版本匹配 |

---

## License / 许可证

[MIT](LICENSE)
