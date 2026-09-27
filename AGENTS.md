# 改这个项目前必读

> 这不是手册，是闸门 —— 下面每条都是踩过才写的。

## 一、动笔前先想清楚「这条该写在哪」

| 你要写的内容 | 写到哪 |
|---|---|
| 怎么用、端口、配置字段、模型参数 | `README.md`（**使用行为的唯一家**） |
| 监控采样、阈值、曲线、共享显存、perf 记录 | `docs\monitor.md` |
| 睡眠 / 关机按钮、远程开机（WOL） | `docs\power.md` |
| 网页绘制、弹层、轮询、验收办法 | `docs\ui.md` |
| 模型参数怎么拼成命令行 | 本目录 `README.md` 第五节 |
| **「这行代码为什么这么写」** | **代码注释** ← 优先放这儿 |

**判据：一份知识只放一处。实现类的「为什么」优先放代码注释** —— 改代码时必然要打开它，
不额外付费、不会过期；写进 README 就是第二份副本，一定会先烂。

## 二、动手前必做

1. **先备份再改**：建一个独立目录 `backup\YYYY-MM-DD-v<目标版本>\`，把要改的文件按原
   相对路径放进去。**改前**校验原文件与备份的 MD5 必须一致 —— 改完才补拷等于存了个假的
   rollback 点。
2. **版本号只在 `ui.html` 第一行**（`<!-- UI_VER: -->`）一处定义，Python 负责替换
   `__UI_VER__` 占位符。写两处 → 手机无限刷新。
3. **换行符**：`ui.html` = LF，`model_manager.py` = CRLF。Python 写文件用 `write_bytes()`，
   `write_text()` 会把 LF 悄悄改成 CRLF；比对多行字符串前先归一成 `\n`。
4. **验收**：`python -m py_compile model_manager.py`；纯视觉改动用 pillow 渲成 PNG 自己看图；
   涉及 DOM / 滚动 / 轮询 / 电源的逻辑，在 node 里搭假 DOM 跑一遍断言（见 `docs\ui.md` 第四节）。
   改完配置相关的代码，起一个**换过端口的临时实例**验一遍真启动，别只做静态检查。

## 三、改了这里就得同步改那几处

| 改了 | 同步改 |
|---|---|
| `ui.html` 的阈值常量（`TEMP_WARN_C` / `POWER_WARN_W` / `MEM_WARN_GB` / `TEMP_MAX_C`） | `docs\monitor.md` 阈值表 + `README.md` |
| 新增 / 改名一个 API 或页面按钮 | `docs\` 对应分册 |
| `config.json` 的字段 | `config.example.json` + `README.md` 第四节 |
| `models.json` 字段 | `README.md` 第五节 |
| 端口 / CLI 参数 | `README.md` 第三节 |

## 四、禁止搜索这些目录

`backup\`（历史版本，**只用于回滚**）、`__pycache__\`、`logs\engine-*.log`、`logs\*.log`
（每个几十 KB 到几百 KB，grep 横扫一次比通读全部文档还贵）。要参考历史实现就**按文件名精确取**。

## 五、平台注意（Windows）

- 控制台是 **GBK**：脚本里别 `print('✔')` 这类字符，会 `UnicodeEncodeError` 把后半段输出吞掉。
  用 `PYTHONIOENCODING=utf-8` + `python -u`。
- **Windows PowerShell 5.1 把无 BOM 的 `.ps1` 按系统 ANSI 读**（中文系统上是 GBK），
  会把源码里的中文字符串弄断，报 `TerminatorExpectedAtEndOfString`。
  **`.ps1` 文件里一律只用 ASCII。**
- `.bat` 里的 `echo` 不要出现 `->`：cmd 当重定向符，会在当前目录吐出垃圾文件并截断输出。
  写完 `grep -n '\->' *.bat` 过一遍。延时用 `ping -n 4 127.0.0.1 >nul`，
  **不要用 `timeout /t`**（Git Bash 会把它换成 GNU 版，报错后延时根本没生效）。
- `taskkill` 返回成功 **≠ 进程已经消失**：它还会在 `tasklist` 里待约 0.5 秒。
  别在这段窗口里判断「还有没有别的实例在跑」，会把刚杀掉的自己认成别人。
  要确定就 `proc.wait(timeout=5)`。
