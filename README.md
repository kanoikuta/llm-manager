# 本地模型管理器（Local LLM Manager）

给 [llama.cpp](https://github.com/ggml-org/llama.cpp) 套一层常驻服务：**网页上切模型、改参数**，
手机浏览器就能用，同时对外暴露 OpenAI 兼容 API。

- **纯标准库 Python，零依赖**（Python 3.10+）
- **一个端口同时是控制台和 API**：`/` 是网页，`/v1` 是 OpenAI 接口
- **按需加载（JIT）**：请求里写哪个模型就自动切哪个；显存只装得下一个时也不用手工腾地方
- **改动即时生效**：参数在网页上改，保存后立刻按新参数重载
- Windows 上还能顺手管一下**睡眠 / 关机**和**显存/功耗曲线**

---

## 一、需要准备什么

| | |
|---|---|
| Python | 3.10 或更高，加进 PATH（或用 `PYTHON` 环境变量指定，见 `start.bat`） |
| 引擎 | llama.cpp 的 **`llama-server.exe`**（CUDA 版才能用显卡） |
| 模型 | 任意 `.gguf` |

**目录结构**（推荐，省得配路径）：

```
<root>\
  bin\llama-server.exe     引擎放这儿
  templates\...            可选：自定义聊天模板
  models\                  模型放这儿（默认扫描目录）
  manager\                 本项目
```

不想这么摆也行 —— 在 `config.json` 里用 `engine` 指绝对路径即可。

## 二、跑起来

1. `llama-server.exe` 放进 `<root>\bin\`，模型 `.gguf` 放进 `<root>\models\`
2. 双击 **`start.bat`**（有窗口，能看日志）
3. 浏览器打开 `http://127.0.0.1/`

首次运行会自动生成一把随机 **API key**，写进 `config.json`，并在控制台打印出来。
网页打开时会问你要这个 key（也可以直接用 `http://<host>/?k=<key>` 进去）。
API 请求带 `Authorization: Bearer <key>`。

开机自启：跑一次 **`install-autostart.bat`**（往「启动」文件夹放一个 `.vbs`，用 `pythonw.exe`
无窗口拉起）。取消就 `remove-autostart.bat`。

## 三、端口

| 端口 | 用途 |
|---|---|
| **80** | 统一入口：`/` 控制台，`/v1` OpenAI API |
| **8091** | 兼容入口（代理），给改不了地址的老客户端用 |
| **8092** | `llama-server` 实际监听，**仅本机**，别对外暴露 |

可以用 `--console-port` / `--proxy-port` / `--engine-port` 改。
`--takeover` 会在启动时杀掉所有 `llama-server` 并接管 8091（专门用来清场）。

> **为什么 8091 是代理而不是让引擎直接监听**：① 客户端配置一个字都不用改；
> ② 可以按需自动加载。代价是切换模型要 30~60 秒（显存只装得下一个，必须先杀后启），
> 这段时间请求会被挂住，成不成取决于你客户端的超时设置。

## 四、配置

复制 `config.example.json` 成 **`config.json`** 再改（后者在 `.gitignore` 里，不会被提交）。
**不复制也能跑**，全部走默认值。字段见模板里的 `_说明`。

| 字段 | 留空时的默认值 |
|---|---|
| `api_key` | 首次运行随机生成一把 |
| `engine` | 管理器上一级的 `bin\llama-server.exe` |
| `chat_template` | 上一级的 `templates\Qwen-Fixed-Chat-Templates\chat_template.jinja` |
| `lan_hostname` | `localhost` |
| `model_dirs` | 管理器同级的 `models\` |
| `presets` | 空 |

**模型参数**在网页上改（模型卡片右边的 ⚙），改动写进 `models.json`，重启不丢。
`models.json` 是**用户数据**，同样不进版本库；首次运行会按 `presets` 生成一份。

## 五、模型参数怎么填

| 字段 | 说明 |
|---|---|
| 上下文长度 ctx | 页面以 k 为单位（1k = 1024 tokens）。**占用显存的大头，先动它** |
| MTP 投机解码 | `内置` = 模型自带 NextN 头（自动识别，一般不用动）；`外挂 draft` = 用同目录的 draft 侧车；`关闭` |
| MTP 深度 nmax / min | nmax 控制候选数，p-min 控制最低接受概率；只在启用 MTP 时传给引擎 |
| KV 缓存类型 | 量化 KV 省显存。`q4_0` 通常能省一半以上，且实测解码速度不掉 |
| 聊天模板 | `auto` / `custom` / `model`。`custom` 用 `chat_template` 指定的 jinja |
| 多模态投影 mmproj | 同目录有 `mmproj-*.gguf` 就自动挂上；**清空 = 这个模型不要视觉** |
| 专家层数 `-ncmoe` | **只有 MoE 模型才显示**。填 N = 前 N 层的专家权重留 CPU，显存不够时装更大 MoE 用，代价是慢 |
| 额外参数 | 原样追加给 `llama-server`，空格分隔，例如 `--threads 8` |
| 系统提示词 | 清空 = 不注入 |

> **经验值**（在 24GB 显存 + 27B 模型上实测）：上下文超过 **80k** 就会溢出到系统内存，
> 解码速度从约 115 t/s 掉到约 26 t/s。显存不够时**先降 ctx，再考虑量化 KV**。
> 同时只能跑一个模型 —— 两个实例并存会互相挤爆显存，速度断崖式下跌。

MoE / dense 是自动识别的（读 gguf 元数据）。认错了可以在 `models.json` 里手写
`"moe": true/false` 纠正 —— **手写的优先，扫描结果不覆盖它**。

## 六、手机 / 局域网访问

1. 跑一次 **`setup-lan.ps1`**（需管理员）：把当前网卡的网络配置设为「专用」，
   开 80/8091 的入站规则（仅本子网），可选把计算机改名以便发布 `<名字>.local`
   ```powershell
   .\setup-lan.ps1 -AdapterMatch 'Wi-Fi' -HostName llama
   ```
2. 跑一次 **`open-firewall.bat`**（需管理员）：补一条 `profile=any` 的端口规则。
   Windows 弹窗自动建的那条**只覆盖 Public 配置文件**，网络被判成「专用」时就不生效了。
3. 手机浏览器打开 `http://<电脑IP>/?k=<key>`

> 装了 [Bonjour](https://support.apple.com/kb/DL999) 才能用 `<名字>.local` 访问；
> 没装就用 IP。

## 七、排查

| 现象 | 原因 / 处置 |
|---|---|
| **手机页面无限刷新** | `ui.html` 第一行的 `UI_VER` 和 Python 返回的对不上。版本号**只在 `ui.html` 第一行定义一处** |
| **端口被占起不来** | `netstat -ano \| findstr :8091` 看是谁，通常是自己手动跑的 `llama-server`。8091 被占时管理器**不抢**，会以「只读控制台」模式启动 |
| **「检测到 N 个不是本管理器启动的 llama-server」** | 有人手动双击了引擎。显存只够一个，管理器**不会去杀**（怕打断正在进行的对话），所以拒绝加载。要收编就在网页上点「接管」 |
| **卸载后进程仍占显存、页面显示空闲** | Windows 的 `tasklist` 对高权限 `llama-server` 可能返回 Access denied。管理器会回退到按 8092 的监听 PID 识别 |
| **一次失败后页面一直卡在「出错」** | `phase` 是**粘**的，没有巡查线程去清它，只有下一次「加载/卸载」才会改写。重试一次即可 |
| 日志 | `logs\manager.log`（管理器自己）、`logs\engine-<id>.log`（引擎输出） |

## 八、改代码

改之前先看一眼 `AGENTS.md`（开发约定 + 文档该写在哪）。
`docs\` 下是分册实现说明：`monitor.md`（监控/曲线）、`power.md`（睡眠/关机/WOL）、`ui.md`（网页）。

三条最容易踩的：

1. **版本号只在 `ui.html` 第一行**（`<!-- UI_VER: -->`）。写两处 → 手机无限刷新。
2. **换行符**：`ui.html` = LF，`model_manager.py` = CRLF。用 Python 写文件请用 `write_bytes()`，
   `write_text()` 会把 LF 悄悄换成 CRLF。
3. **`config.json` 里的东西不要写进源码** —— 源码里不留任何本机信息。

## 许可证

MIT，见 `LICENSE`。
