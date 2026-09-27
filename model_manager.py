#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型管理器 (Local LLM Manager)
==================================

手机上就能切换电脑上的大模型，像 LM Studio 那样。

端口分工
--------
  80    统一局域网入口            / = 网页控制台，/v1 = OpenAI API
  8091  兼容 OpenAI 入口（代理）  旧客户端仍可继续使用
  8092  llama-server 实际监听     内部端口，只有管理器连

为什么 8091 是代理而不是让 llama-server 直接监听？两个好处：
  1. 手机上前端的配置一个字都不用改（原来就是 8091）
  2. 可以按需自动加载（JIT）：请求哪个模型就自动切哪个

JIT 的代价：切换要 30~60 秒（显存只装得下一个模型，必须先杀后启）。
这段时间请求会被挂住不响应，能不能成功取决于客户端的超时设置。

硬约束（来自 AGENTS.md，别绕）
------------------------------
  - 24G 显存同时只能跑一个模型，两台并存会溢出到系统内存，速度 115 -> 26
  - 所以切换必然是「杀旧的 -> 等显存回落 -> 启新的」，中间断服务是正常的

用法
----
  双击 start.bat                   前台启动，能看日志
  install-autostart.bat            装开机自启
  手机浏览器打开 http://<lan_hostname>/?k=<API_KEY>     （两个值都在 config.json 里）
"""

from __future__ import annotations

import argparse
import ctypes
import http.client
import http.server
import json
import os
import re
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ────────────────────────────── 配置 ──────────────────────────────

BASE = Path(__file__).resolve().parent
LLAMA = BASE.parent                      # 上一级目录（引擎和模板默认放在那儿）
LOGDIR = BASE / "logs"
REGISTRY = BASE / "models.json"
USAGE_FILE = BASE / "model_usage.json"
PERF_FILE = LOGDIR / "inference-history.jsonl"
PERF_LOCK = threading.Lock()


# log() 定义在这里（而不是原来的「小工具」段）是因为下面读 config.json 时要用它，
# 而调用发生在模块加载期 —— 放后面会 NameError。
def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    try:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        with open(LOGDIR / "manager.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── config.json：本机配置，不进版本库 ─────────────────────────────────
# 源码里不留任何本机信息：API key、模型目录、主机名、预设参数全从这儿读。
# 公开仓库里只有 config.example.json；本机的 config.json 写在 .gitignore 里。

CONFIG_FILE = BASE / "config.json"


def load_config() -> dict:
    """读 config.json。没有、读不动、格式坏了 —— 一律当空字典，全部走默认值照样能跑。"""
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


CFG = load_config()

# 引擎和聊天模板：默认在管理器的上一级（<root>\bin\llama-server.exe、
# <root>\templates\...\chat_template.jinja）。目录结构不一样就在 config.json 里
# 用 engine / chat_template 指过去 —— 这两个是唯一必须存在的路径，其余都有默认值。
ENGINE = Path(str(CFG.get("engine") or (LLAMA / "bin" / "llama-server.exe")))
TEMPLATE = Path(str(CFG.get("chat_template")
                     or (LLAMA / "templates" / "Qwen-Fixed-Chat-Templates" / "chat_template.jinja")))


def _cfg_paths(key: str, default: list[Path]) -> list[Path]:
    """读一个「路径数组」配置项；没配或配坏了就用默认值。"""
    v = CFG.get(key)
    if not isinstance(v, list) or not v:
        return default
    return [Path(str(p)) for p in v]


def _ensure_api_key() -> str:
    """API key：先读 config.json，没有就随机生成一把写进去。

    为什么不干脆每次启动随机一把：这个 key 会出现在客户端 URL 里（?k=xxx），
    换掉就等于手机书签、客户端配置全部失效 —— 所以生成一次、落盘、以后一直用。
    token_hex(24) = 48 个十六进制字符，长度和以前那个写死的 key 一致。
    """
    key = str(CFG.get("api_key") or "").strip()
    if key:
        return key
    key = secrets.token_hex(24)
    CFG["api_key"] = key
    try:
        CONFIG_FILE.write_text(
            json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"已生成 {CONFIG_FILE.name}，随机 API key 写进去了")
    except OSError as e:
        # 写不进去也不能让管理器起不来：本次用这把临时 key，重启会变
        log(f"[warn] 写不了 {CONFIG_FILE.name}: {e}；本次是临时 key，重启后会变")
    return key

def load_perf_history(limit: int = 50) -> list[dict]:
    rows = []
    try:
        with PERF_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except ValueError:
                    continue
    except OSError:
        return []
    return rows[-limit:]

def load_latest_perf() -> dict[str, dict]:
    latest = {}
    try:
        with PERF_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("model"):
                        latest[str(row["model"])] = row
                except ValueError:
                    continue
    except OSError:
        pass
    return latest

def append_perf_history(row: dict) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with PERF_LOCK:
        with PERF_FILE.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()

# ────────────────────── t/s 摘要（不同配置下的速度） ──────────────────────
# 用户 2026-09-26 定的规矩，三句话：
#   ①「记录最好是程序自动写出结果，不要让你（AI）修改写上去这种」—— 这段文本由**管理器
#     自己**算出来、自己落盘，人不许手改；页面只负责原样显示。
#   ②「日志只写同一设置下 t/s」「同样设置只有 MTP 区别，只写 MTP3 90t/s MTP4 92t/s」——
#     同一设置取平均；同一族里每条都一样的字段不重复写。
#   ③「详细平均速度只写在日志里作为 ai 看的，避免反复测速」—— 这段 + 逐次明细
#     (inference-history.jsonl) 就是给 AI 查的，要比参数先读这两份，别重测。
# 为什么放在 Python 而不是页面里：页面算的话，「谁写的数」就又说不清了 —— 而且 AI
# 读不到浏览器里的 JS 结果，却能直接读 tps-summary.md。
TPS_SUMMARY_FILE = LOGDIR / "tps-summary.md"
TPS_HEADER = ("# t/s 摘要 · 管理器自动生成（别手改）· 同一设置取平均，括号里是次数 · "
              r"逐次明细 logs\inference-history.jsonl")


def compact_ctx(ctx) -> str:
    """上下文按 k 显示，和页面「参数」里 1k = 1024 tokens 是同一套。"""
    if not ctx:
        return ""
    k = ctx / 1024
    s = str(int(k)) if float(k).is_integer() else f"{k:.3f}".rstrip("0").rstrip(".")
    return s + "k"


def _dwidth(s: str) -> int:
    """等宽字体下的显示宽度：CJK / 全角标点占**两列**。

    对齐必须按这个算 —— 用 len() 的话「无 MTP」「配置未记录」会把整列推歪。
    范围取自 Unicode 的常见宽字符区（终端 wcwidth 那一套的简化版，够用了）。
    """
    w = 0
    for ch in s:
        o = ord(ch)
        if (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3
                or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE6F
                or 0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6):
            w += 2
        else:
            w += 1
    return w


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _mtp_label(v: dict, vary_nmax: bool, vary_pmin: bool) -> str:
    """变体标签：**只写有区别的那一档**（用户要的就是「MTP 3 / MTP 4」这种）。"""
    if v["mtp"] == "?":
        return "MTP 未记录"          # 老记录没存这个字段，不臆造「无 MTP」
    if v["mtp"] == "none":
        return "无 MTP"              # 明确写着 none 才是真的没开
    if vary_nmax and vary_pmin:
        return f"MTP {v['nmax']}/{v['pmin']}"
    if vary_nmax:
        return f"MTP {v['nmax']}"
    if vary_pmin:
        # 别缩成「MTP 0.7」—— 那读起来像 nmax
        return f"MTP p-min {v['pmin']}"
    return f"MTP {v['nmax']}/{v['pmin']}"


def build_tps_summary(rows) -> str:
    """把推理记录汇总成一段文本（页面末段和 tps-summary.md 共用这一份，别各写一份）。

    分组三级：**模型 → 上下文 · KV → MTP 变体**。模型必须先分：跨模型比 t/s 没有意义。
    列按显示宽度对齐、一行一条。老记录（v0.3.05 之前）没有 ctx/kv/mtp，
    写「配置未记录 / MTP 未记录」，不拿别的记录去凑。
    """
    models: dict = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or r.get("model") or "（未知模型）"
        base = " · ".join(x for x in (compact_ctx(r.get("ctx")), r.get("kv") or "") if x) \
               or "配置未记录"
        pmin = 0 if r.get("pmin") is None else r.get("pmin")
        nmax = "" if r.get("nmax") is None else str(r.get("nmax"))
        key = (r.get("mtp") or "?", nmax, pmin)
        models.setdefault(name, {}).setdefault(base, {}).setdefault(key, []).append(r)

    def last_at(bases) -> str:
        return max((str(r.get("at") or "")
                    for variants in bases.values()
                    for recs in variants.values() for r in recs), default="")

    blocks = []
    for name, bases in sorted(models.items(), key=lambda kv: last_at(kv[1]), reverse=True):
        flat = []                     # (base, label, avg, n) —— 先摊平，才知道每列多宽
        for base, variants in bases.items():
            vs = []
            for (mtp, nmax, pmin), recs in variants.items():
                tps = [x["tps"] for x in recs if isinstance(x.get("tps"), (int, float))]
                if tps:
                    vs.append({"mtp": mtp, "nmax": nmax, "pmin": pmin, "tps": tps})
            vs.sort(key=lambda v: (_num(v["nmax"]), _num(v["pmin"])))
            if not vs:
                continue
            vary_nmax = len({v["nmax"] for v in vs}) > 1
            vary_pmin = len({v["pmin"] for v in vs}) > 1
            for v in vs:
                flat.append((base, _mtp_label(v, vary_nmax, vary_pmin),
                             sum(v["tps"]) / len(v["tps"]), len(v["tps"])))
        if not flat:
            continue
        wb = max(_dwidth(b) for b, _, _, _ in flat)
        wl = max(_dwidth(l) for _, l, _, _ in flat)
        lines = ["  " + b + " " * (wb - _dwidth(b)) + "   "
                 + l + " " * (wl - _dwidth(l)) + "   "
                 + f"{a:6.1f} t/s" + "   " + f"{n} 次"
                 for b, l, a, n in flat]
        blocks.append(name + "\n" + "\n".join(lines))

    if not blocks:
        return TPS_HEADER + "\n\n（还没有记录 —— 聊一次就有了）\n"
    return TPS_HEADER + "\n\n" + "\n\n".join(blocks) + "\n"


def write_tps_summary(rows) -> str:
    """算出来 + 写进 logs\\tps-summary.md，返回文本（页面那次直接用返回值，不重读文件）。"""
    text = build_tps_summary(rows)
    try:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        tmp = TPS_SUMMARY_FILE.with_suffix(".tmp")
        # 用 write_bytes：write_text 不显式给 newline 时会把 \n 翻成 \r\n（AGENTS.md 第二节）
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, TPS_SUMMARY_FILE)
    except OSError as e:
        log(f"t/s 摘要写入失败: {e}")
    return text


def load_model_usage() -> dict[str, int]:
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): max(0, int(v)) for k, v in data.items()}
    except (OSError, ValueError, TypeError):
        pass
    return {}

def save_model_usage(counts: dict[str, int]) -> None:
    tmp = USAGE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, USAGE_FILE)

# 模型目录：读 config.json 的 model_dirs；没配就找管理器同级的 models\ 目录
MODEL_DIRS = _cfg_paths("model_dirs", [BASE / "models"])

API_KEY = _ensure_api_key()
LAN_HOSTNAME = str(CFG.get("lan_hostname") or "localhost")  # mDNS 名；Bonjour 随 Wi-Fi 地址变化更新记录

MANAGER_PORT = 80        # / 显示控制台，/v1 由代理处理
PROXY_PORT = 8091        # 对外 OpenAI 入口
ENGINE_PORT = 8092       # llama-server

VRAM_IDLE_MB = 3000      # 显存回落到这个值以下才算上一个模型释放干净
VRAM_TOTAL_MB = 24455
VRAM_TIMEOUT = 40        # 等显存回落的最长秒数
LOAD_TIMEOUT = 240       # 等模型就绪的最长秒数

# 自己刚杀掉的 PID 在这段时间内不算「外部进程」（为什么需要：见 external_pids）。
OWN_DEAD_GRACE = 20.0

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

PROXY_OK = False         # 8091 代理是否真的起来了（被别的 llama-server 占着时是 False）

# 网页拆到单独的文件了。版本号写在 ui.html 第一行的注释里 —— 只改一个地方，
# Python 这边自动读出来。改了页面直接刷新就生效，不用重启管理器。
UI_FILE = BASE / "ui.html"

_ui_cache: dict = {"text": "", "ver": "0", "mtime": 0.0}


def load_ui() -> tuple[str, str]:
    """返回 (页面 HTML, 版本号)。

    按文件 mtime 缓存，所以编辑 ui.html 后刷新页面就能看到，不用重启。
    手机上的旧页面靠版本号发现不一致，自己重载。
    """
    try:
        mt = UI_FILE.stat().st_mtime
        if _ui_cache["mtime"] != mt:
            text = UI_FILE.read_text(encoding="utf-8")
            m = re.search(r"UI_VER:\s*([\w.]+)", text)
            ver = m.group(1) if m else "0"
            # 把页面里的 __UI_VER__ 换成真实版本号 —— 这样版本只在注释里定义一处。
            # 之前是手写两份，拆分时漏改了一份，两边永远对不上，手机无限刷新。
            text = text.replace("__UI_VER__", ver)
            _ui_cache.update(text=text, ver=ver, mtime=mt)
    except Exception as e:
        if not _ui_cache["text"]:
            _ui_cache["text"] = (
                "<!doctype html><meta charset='utf-8'>"
                "<body style='font:15px sans-serif;background:#0f1115;color:#e6e9ef;padding:24px'>"
                f"<h3>读不到 ui.html</h3><p>{e}</p>"
                f"<p>应该在：<code>{UI_FILE}</code></p>")
    return _ui_cache["text"], _ui_cache["ver"]

# 常用预设：从 config.json 的 presets 读（本机实测的推理参数写在那儿）。
# 没配就是空表 —— 模型列表完全由 model_dirs 扫描得出，不影响使用。
PRESETS = [p for p in (CFG.get("presets") or [])
           if isinstance(p, dict) and p.get("id") and p.get("file")]

# 默认加载哪个：本机的老行为一直是「预设里的第一个」，所以默认就照这个取
DEFAULT_ID = str(CFG.get("default_id") or (PRESETS[0]["id"] if PRESETS else ""))

# 这些模型需要打过 HauhauCS 补丁的引擎才能跑，官方引擎下会失败
BROKEN_HINT = {
    "fastmtp": "FastMTP 侧车需要打过补丁的引擎（当前 bin\\ 是官方版），加载会失败或没收益",
}


# ────────────────────────────── 小工具 ──────────────────────────────

def now() -> float:
    return time.time()


def tail_manager_log(n: int = 250) -> list[str]:
    """manager.log 的最后 n 行，给网页的「日志」弹层用。

    和 `RT.load_log` 是两份东西，别混：那个只记模型加载/卸载（含引擎命令行、
    显存等待），这个记的是管理器自己的运行日志（睡眠、关机、改参数）。
    网页上分两段显示，各看各的。

    只读文件尾巴 64KB：整份读进来的话，跑几天就是几百 KB 灌到手机上。
    日志读不到就返回空列表 —— 看不了日志不该让整个接口报错。
    """
    try:
        with open(LOGDIR / "manager.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            data = f.read()
        # 从中间截断时第一行是半行，但 64KB 远多于 n 行，[-n:] 会把它挤掉
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def vram_used_mb() -> int:
    """当前显存占用（MB），失败返回 -1"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def mem_used_mb() -> tuple[int, int]:
    """系统内存 (已用MB, 总MB)。

    直接调 GlobalMemoryStatusEx —— 比跑 wmic/PowerShell 子进程快几个数量级，
    每 1.5 秒轮询一次也不心疼。失败返回 (-1, -1)。
    """
    try:
        st = _MEMSTATUS()
        st.dwLength = ctypes.sizeof(_MEMSTATUS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return (-1, -1)
        total = st.ullTotalPhys // (1024 * 1024)
        avail = st.ullAvailPhys // (1024 * 1024)
        return (total - avail, total)
    except Exception:
        return (-1, -1)


def kill_engines() -> int:
    """杀掉所有 llama-server。**只在「接管」时调用**。

    日常切换走 Runtime._kill_proc()，只杀自己启动的那个 —— 不能无脑杀：
    用户很可能正用着手动启动的模型在聊天，杀了就把他打断了。
    """
    n = 0
    for exe in ("llama-server.exe", "llama-cli.exe", "llama-bench.exe"):
        try:
            r = subprocess.run(["taskkill", "/F", "/IM", exe],
                               capture_output=True, text=True, timeout=15,
                               creationflags=CREATE_NO_WINDOW)
            n += r.stdout.count("SUCCESS")
        except Exception:
            pass
    return n


def engine_pids() -> list[int]:
    """当前系统上能识别到的 llama-server PID。

    tasklist 在进程属于提升权限的会话时可能只返回 Access denied；不能把
    查询失败误当作进程已退出，否则卸载会清空管理器状态但留下模型占显存。
    至少用内部引擎端口的监听 PID 做兜底。
    """
    pids = []
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW)
        for line in out.stdout.splitlines():
            parts = [x.strip().strip('"') for x in line.split(",")]
            if len(parts) >= 2 and parts[0].lower().startswith("llama-server"):
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    pass
    except Exception:
        pass
    if not pids and port_busy(ENGINE_PORT):
        pid = pid_on_port(ENGINE_PORT)
        if pid:
            pids.append(pid)
    return list(dict.fromkeys(pids))


def pid_on_port(port: int) -> int | None:
    """哪个进程在监听这个端口"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=CREATE_NO_WINDOW)
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return int(parts[4])
    except Exception:
        pass
    return None


def engine_alias() -> str:
    """问 8092 上的引擎：你的别名是什么。

    启动时给它带了 `-a <模型id>`，所以 /v1/models 会把 id 报出来。
    比读进程命令行可靠得多 —— **wmic 在新版 Windows 11 上已经被移除了**，
    之前用 wmic 读命令行，结果什么都读不到，认领一直失败。

    别人手动跑的 llama-server 不带 -a，报回来的会是模型文件路径，认不出来。
    """
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=4)
        conn.request("GET", "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
        d = json.loads(conn.getresponse().read())
        conn.close()
        items = d.get("models") or d.get("data") or []
        if items:
            it = items[0]
            return (it.get("id") or it.get("model") or it.get("name") or "").strip()
    except Exception:
        pass
    return ""


def engine_healthy(timeout: float = 4.0) -> bool:
    """8092 上的引擎健康吗（/health 返回 ok）"""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=timeout)
        conn.request("GET", "/health", headers={"Authorization": f"Bearer {API_KEY}"})
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return r.status == 200 and b"ok" in body.lower()
    except Exception:
        return False


class _AdoptedProc:
    """认领过来的引擎：不是本进程启动的，但归我们管。

    只实现 Popen 被用到的那几个接口（pid / poll / wait / kill），
    这样 Runtime 里原有的启停逻辑一行都不用改。
    """

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        return None if self.pid in engine_pids() else 0

    def wait(self, timeout=None):
        t0 = now()
        while self.poll() is None:
            if timeout is not None and now() - t0 > timeout:
                raise subprocess.TimeoutExpired("adopted", timeout)
            time.sleep(0.3)
        return 0

    def kill(self):
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.pid)],
                           capture_output=True, text=True, timeout=15,
                           creationflags=CREATE_NO_WINDOW)
        except Exception:
            pass


def slugify(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "-", s).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)


def human_gb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 ** 3), 2)
    except Exception:
        return 0.0


def is_qwen(path: Path) -> bool:
    return "qwen" in str(path).lower()


# ── GGUF 头部元数据：用来补作者信息 ──
# 实测大部分量化作者不写 general.author（17 个模型里只有 HauhauCS 写了），
# 所以这只是目录结构之外的兜底。

_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_WANT = ("general.author", "general.creator", "general.organization",
              "general.architecture", "general.basename")
# 整数类型。nextn_predict_layers 是层数，只认整数，免得把 float 读成层数
_GGUF_INT = (0, 1, 2, 3, 4, 5, 10, 11)
# MTP 有无的唯一判据，键名带架构前缀（qwen35.nextn_predict_layers），所以按后缀认
_NEXTN_KEY = ".nextn_predict_layers"
# MoE 的铁证：专家个数（键名同样带架构前缀，所以按后缀认）。
# 注意它**不进上面那个提前收手的条件** —— 元数据是哈希序，专家数可能排在很后面，
# 为了它把每个 dense 模型的整个元数据（含十几万条的词表）都读一遍不值当（见 gguf_meta）。
_EXPERT_KEY = ".expert_count"


def _gguf_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    if n > 200_000_000:            # 指针一旦错位就会读到这种垃圾长度，直接拦掉
        raise ValueError(f"字符串长度不合理: {n}")
    return f.read(n).decode("utf-8", "replace")


_STRARRAY_CHUNK = 1 << 20


def _gguf_skip_strs(f, n: int) -> None:
    """跳过 n 个字符串 —— 给词表那种十几万条的数组用。

    以前这里直接抛「数组太大，放弃」，赌的是「要的字段都在最前面几条」。
    **这个赌注是错的**：llama.cpp 写元数据的顺序是哈希序，跟语义无关。实测
    NVFP4 的 `tokenizer.ggml.tokens` 排在第 2 条、`qwen35.nextn_predict_layers` 排在第 26 条，
    于是读到词表就收手，MTP 永远判成「没有」（Omega 恰好反过来，所以当时没露馅）。

    也不能逐个 read+seek：十几万次系统调用太慢。按块读进来，在内存里数长度前缀。
    """
    buf = b""
    pos = 0
    for _ in range(n):
        while len(buf) - pos < 8:                  # 先凑够 8 字节的长度前缀
            chunk = f.read(_STRARRAY_CHUNK)
            if not chunk:
                raise ValueError("文件提前结束")
            buf = buf[pos:] + chunk                # 只在换块时搬一次，别每轮都切
            pos = 0
        ln = struct.unpack_from("<Q", buf, pos)[0]
        if ln > 200_000_000:
            raise ValueError(f"字符串长度不合理: {ln}")
        pos += 8
        while len(buf) - pos < ln:                 # 再凑够这么长的内容
            chunk = f.read(max(_STRARRAY_CHUNK, ln - (len(buf) - pos)))
            if not chunk:
                raise ValueError("文件提前结束")
            buf = buf[pos:] + chunk
            pos = 0
        pos += ln
    if len(buf) > pos:                             # 多读进来的还回去
        f.seek(pos - len(buf), 1)


def _gguf_skip(f, t: int) -> None:
    if t == 8:
        _gguf_str(f)
    elif t == 9:                                   # 数组
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et in _GGUF_FIXED:
            # 定长元素可以直接 seek。词表那种十几万条的数组逐个读会慢死
            f.seek(n * _GGUF_FIXED[et], 1)
        elif et == 8:
            # 变长元素只有字符串这一种，走按块走字的快路
            _gguf_skip_strs(f, n)
        else:
            for _ in range(n):
                _gguf_skip(f, et)
    elif t in _GGUF_FIXED:
        f.read(_GGUF_FIXED[t])
    else:
        raise ValueError(f"未知 gguf 类型 {t}")


def gguf_meta(path: Path) -> dict:
    """从 gguf 头部读作者 / 架构 / 基座名 / NextN 层数。读不到就是空 dict，不抛异常。

    元数据都在文件最前面，读几 KB 就够，不用把 15GB 全扫一遍。

    架构字段很有用：Dark-Scarlett、Omega-Convergence、Serenity 这些
    文件名里没有 "qwen"，但 architecture 是 qwen35 —— 是 Qwen3.8 的微调，
    该套自定义聊天模板。只看文件名会把它们漏掉。

    nextn 同理且更极端：MTP 头在不在，文件名完全说了不算（见 discover_models）。
    判据就是元数据里的 `qwen35.nextn_predict_layers`，不用去扫张量表（那张表在词表后面，
    为了读到它得把整个元数据趟一遍）。
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            f.read(4)                              # version
            f.read(8)                              # tensor count
            n_kv = struct.unpack("<Q", f.read(8))[0]
            found = {}
            nextn = None                           # None = 还没读到那个键（0 也是有值的）
            experts = None                         # 同上：专家个数，读到就是 MoE 的铁证
            for _ in range(min(n_kv, 400)):
                try:
                    k = _gguf_str(f)
                    t = struct.unpack("<I", f.read(4))[0]
                    if k in _GGUF_WANT and t == 8:
                        found[k] = _gguf_str(f)
                        # 两个条件都满足才收工。键序是哈希序、跟语义无关，
                        # 凑齐 author 就 break 的话 nextn 可能还在后面没读到
                        if len(found) >= len(_GGUF_WANT) and nextn is not None:
                            break
                    elif k.endswith(_NEXTN_KEY) and t in _GGUF_INT:
                        nextn = int.from_bytes(f.read(_GGUF_FIXED[t]), "little")
                    elif k.endswith(_EXPERT_KEY) and t in _GGUF_INT:
                        # 顺路认一下，读不到也不影响（判 MoE 主要靠 arch，见 is_moe）
                        experts = int.from_bytes(f.read(_GGUF_FIXED[t]), "little")
                    else:
                        _gguf_skip(f, t)
                except Exception:
                    # 读不动了（指针错位 / 遇到没见过的类型）。拿着已经读到的走人 ——
                    # 以前这里不 catch，结果指针错位后抛 MemoryError，整个函数白读，
                    # architecture 明明在第 0 条。**这层兜底别去掉。**
                    break
            return {
                "author": (found.get("general.author")
                           or found.get("general.creator")
                           or found.get("general.organization") or ""),
                "arch": found.get("general.architecture", ""),
                "base": found.get("general.basename", ""),
                "nextn": nextn or 0,
                "experts": experts or 0,
            }
    except Exception:
        return {}


def is_moe(meta: dict) -> bool:
    """这个 gguf 是 MoE 还是 dense —— 决定参数页要不要给「专家层数」那个框。

    主判据是 architecture：llama.cpp 的 MoE 架构名基本都带 moe（qwen35moe / hunyuan-moe /
    exaone-moe / mixtral…，本机的 Cyber-Tiel 35B-A3B、Ornith-35B-A3B 都是 qwen35moe）。
    **不带的也有**（deepseek2、llama4、gpt-oss…），所以再看一眼 `<arch>.expert_count`：
    它不一定能在提前收手之前读到（元数据哈希序），**读到了就是铁证**，读不到也不冤枉 dense。

    误判成 dense 的代价 = 少一个可填的框，不影响加载；误判成 MoE 才会往命令行里塞
    `-ncmoe`，那对 dense 模型是无效参数。所以两个条件里只认**确凿**的那个。
    """
    if "moe" in str(meta.get("arch") or "").lower():
        return True
    return int(meta.get("experts") or 0) > 0


def detect_author(p: Path, root: Path) -> str:
    """从目录结构提作者。

        lms\\models\\JonathanColetti\\Qwen3.8-27B-Uncensored-GGUF\\x.gguf  -> JonathanColetti
        lms\\models\\unsloth\\Qwen3.8-27B-GGUF\\x.gguf                     -> unsloth
        models.gguf\\x.gguf                                            -> （没有作者层级）
    """
    rel = p.relative_to(root).parts
    return rel[0] if len(rel) >= 2 else ""


# ────────────────────────── 模型发现 ──────────────────────────

def discover_models() -> dict:
    """扫描模型目录，合并 models.json 里的预设。

    过滤规则：
      - mmproj*.gguf 是多模态投影副文件，不是模型本体
      - *draft*.gguf 是投机解码侧车，会被自动挂到同目录的主模型上
      - mmproj 同样按侧车处理，自动挂到同目录的主模型上（见 build_cmd）
    """
    models: dict[str, dict] = {}

    # 1) 先收集候选主模型文件
    #    过滤掉不是模型本体的东西：
    #      mmproj-*.gguf            多模态投影副文件
    #      *draft*.gguf             投机解码侧车
    #      *FastMTP-32K*.gguf       裁剪词表的 MTP 侧车（0.8GB 那个，需要补丁引擎，已放弃）
    files: list[tuple[Path, str, Path]] = []
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.gguf")):
            low = p.name.lower()
            if "mmproj" in low or "draft" in low or "fastmtp" in low:
                continue
            rel = p.relative_to(d).parts
            files.append((p, rel[0] if len(rel) > 1 else "", d))

    # 不同目录里可能有同一份模型，重名的要能分清
    from collections import Counter
    dup = Counter(p.stem for p, _, _ in files)

    for p, label, root in files:
        meta = gguf_meta(p)
        # 作者：先看目录结构（最可靠，用户的库就是这么组织的），
        # 没有作者层级（比如 models.gguf\ 下）再读 gguf 元数据兜底
        author = detect_author(p, root) or meta.get("author", "")

        # MTP 有没有，只认 gguf 里的 NextN 层，**不看文件名** —— 文件名两个方向都会骗人：
        #   ReadyArt 全系（Dark-Scarlett / Omega-Convergence / Serenity）文件名不带 MTP，
        #     实际都自带 NextN 头（27B 那批 866 张量 = 850 + 16 张 MTP）；
        #   DavidAU 的目录名写着 MTP，可那个 NEO 量化里一层都没有（851 张量）。
        # 判反的代价不对称：真有 MTP 却没开 → 引擎报 "wrong number of tensors" 直接加载失败；
        # 没有 MTP 却开了 → 白白多一个空转的投机解码。所以宁可信文件内容。
        mtp = "builtin" if meta.get("nextn") else "none"
        if not meta.get("arch") and "MTP" in str(p).upper():
            mtp = "builtin"                        # 头部压根读不动时的兜底，保持老行为

        # 同目录下的 draft 侧车，自动挂上
        draft = None
        for sib in p.parent.glob("*.gguf"):
            if "draft" in sib.name.lower():
                draft = str(sib)
                mtp = "external"
                break

        # 同目录下的 mmproj 侧车（多模态投影），自动挂上。
        # 不挂的话引擎**完全没有视觉能力** —— 发图会 500
        # "image input is not supported - hint: ... provide the mmproj"。
        # 一个目录通常只有一份 mmproj，所以挑排序后的第一份；
        # 想换/想关就在 models.json 里覆盖或清空 mmproj 字段。
        mmproj = next(
            (
                str(s)
                for s in sorted(p.parent.glob("*.gguf"))
                if "mmproj" in s.name.lower()
            ),
            None,
        )

        # 重名时：有作者标签就够区分了，不用再往名字里塞目录名
        name = p.stem if (dup[p.stem] == 1 or author) else f"{p.stem} ({label})"

        mid = slugify(p.stem)[:64]
        while mid in models:                       # 同名去重
            mid += "-2"

        item = {
            "id": mid,
            "name": name,
            "author": author,
            "arch": meta.get("arch", ""),
            "base": meta.get("base", ""),
            # MoE 还是 dense：参数页据此决定给不给「专家层数」那个框（见 is_moe）
            "moe": is_moe(meta),
            "experts": meta.get("experts", 0),
            "file": str(p),
            "mtp": mtp,
            "ctx": 32768,
            # 有投机解码才谈得上深度；没有的话这个数根本不进命令行。
            # 3 是实测最优点（4 已经转负），自动发现也照这个给
            "nmax": 3 if mtp in ("builtin", "external") else 2,
            "size_gb": human_gb(p),
            "verified": False,
            "note": "自动发现 · 参数保守，未实测",
            "star": False,
        }
        attached: list[str] = []
        if mtp == "builtin":
            attached.append("内置 MTP")             # 卡片上直接能看出谁开了
        if draft:
            item["draft"] = draft
            attached.append("挂外挂 draft")
        if mmproj:
            item["mmproj"] = mmproj
            attached.append("视觉")
        if attached:
            item["note"] = "自动发现 · " + " + ".join(attached)

        for k, hint in BROKEN_HINT.items():
            if k in str(p).lower():
                item["warning"] = hint

        models[mid] = item

    # 2) 预设覆盖（用户点名要的两个，参数写死）
    for preset in PRESETS:
        f = Path(preset["file"])
        # 先按文件路径找到自动扫描出来的那条，把它替换掉，避免重复
        for mid, m in list(models.items()):
            if os.path.normcase(m["file"]) == os.path.normcase(preset["file"]):
                del models[mid]
        preset = dict(preset)
        preset["exists"] = f.exists()
        preset["size_gb"] = human_gb(f)
        preset["verified"] = True
        if f.exists():
            meta = gguf_meta(f)
            preset.setdefault("arch", meta.get("arch", ""))
            preset.setdefault("base", meta.get("base", ""))
            # setdefault：写在 PRESETS 里的值优先，没写才按 gguf 判
            preset.setdefault("moe", is_moe(meta))
            preset.setdefault("experts", meta.get("experts", 0))
        models[preset["id"]] = preset

    # 3) models.json 里的用户覆盖（优先级最高）
    if REGISTRY.exists():
        try:
            user = json.loads(REGISTRY.read_text(encoding="utf-8"))
            for mid, patch in user.get("models", {}).items():
                if mid in models:
                    models[mid].update(patch)
                    models[mid]["verified"] = True
                else:
                    patch.setdefault("id", mid)
                    patch.setdefault("name", mid)
                    patch.setdefault("mtp", "none")
                    patch.setdefault("ctx", 32768)
                    patch.setdefault("nmax", 2)
                    if patch.get("file"):
                        patch["size_gb"] = human_gb(Path(patch["file"]))
                        patch["exists"] = Path(patch["file"]).exists()
                    models[mid] = patch
        except Exception as e:
            log(f"[warn] models.json 读取失败，已忽略: {e}")

    # 4) MoE 标记兜底：上面三条路都可能没判过（直接写进 models.json 的条目、file 是后填的），
    #    这里补一次。**已经有 moe 的不动** —— models.json 里手写的值优先（扫描是猜，手写是准）。
    for m in models.values():
        if "moe" in m:
            continue
        f = m.get("file")
        if not f:
            m["moe"] = False                 # 连文件都没有，谈不上 MoE
            continue
        try:
            meta = gguf_meta(Path(f))
            m["moe"] = is_moe(meta)
            m.setdefault("experts", meta.get("experts", 0))
        except Exception:
            m["moe"] = False

    return models


def write_registry(models: dict) -> None:
    """首次运行时把两个预设写成 models.json，方便用户手改"""
    if REGISTRY.exists():
        return
    doc = {
        "_说明": [
            "改这里的参数会覆盖自动扫描的结果，改完重启管理器生效。",
            "ctx=上下文长度, nmax=投机解码深度, mtp=builtin/external/none",
            "extra 是追加给 llama-server 的额外参数数组",
            "预设模型参数来自本机实测；star 仅控制列表高亮。",
        ],
        "models": {p["id"]: {k: v for k, v in p.items() if k != "id"} for p in PRESETS},
    }
    try:
        REGISTRY.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[warn] models.json 写入失败: {e}")


def build_cmd(m: dict) -> list[str]:
    """把模型配置翻译成 llama-server 命令行"""
    f = Path(m["file"])
    cmd = [str(ENGINE), "-m", str(f)]

    has_draft = bool(m.get("draft") and Path(m["draft"]).exists())
    if has_draft:
        cmd += ["--model-draft", m["draft"]]

    # 多模态投影。没有它引擎对图片一律回
    #   500 "image input is not supported - hint: ... provide the mmproj"
    # 加上之后 /props 的 modalities.vision 才是 true。
    # 显存代价约 0.9GB（BF16 的 27B 投影），只有目录里真有 mmproj 的模型才加。
    if m.get("mmproj") and Path(m["mmproj"]).exists():
        cmd += ["--mmproj", m["mmproj"]]

    # *** 自带 NextN/MTP 头的模型不加这个会报 ***
    #     done_getting_tensors: wrong number of tensors; expected 1866, got 1850
    #     那是「MTP 没开」，不是文件损坏。谁有 NextN 头由 gguf 说了算（见 discover_models）
    #
    # mtp=external 但 draft 文件没了（被删 / 手机上选错档）→ 当作没开：
    # 否则会塞一个没有 draft 的 --spec-type draft-mtp 进去，模型又恰好没有 NextN 头，
    # 加载直接失败，报的还是上面那种看不懂的错
    spec_on = m.get("mtp") == "builtin" or (m.get("mtp") == "external" and has_draft)
    if spec_on:
        cmd += ["--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(m.get("nmax", 3)),
                "--spec-draft-p-min", str(m.get("pmin", 0.0))]

    # 是不是 Qwen3.8 系：看 gguf 的 architecture 最准（qwen35 / qwen35moe）。
    # 光看文件名会漏掉 Dark-Scarlett / Omega-Convergence / Serenity 这类
    # 名字里不带 qwen 的微调版 —— 它们其实都是 qwen35。
    arch = (m.get("arch") or "").lower()
    qwen_like = arch.startswith("qwen") or (not arch and is_qwen(f))

    # chat_template：auto = Qwen 系套自定义模板；custom = 强制套；model = 一律用自带的
    tpl = m.get("template", "auto")
    if (tpl == "custom" or (tpl == "auto" and qwen_like)) and TEMPLATE.exists():
        cmd += ["--chat-template-file", str(TEMPLATE)]

    if m.get("mtp") in ("builtin", "external") or qwen_like:
        cmd += ["--reasoning-format", "deepseek", "--reasoning-preserve"]

    # MoE 专家层数：llama.cpp 的 `-ncmoe N` = 把**前 N 层的专家权重**留在 CPU。
    # 这是引擎里唯一跟「专家」有关的可调项（`--help` 里只有 -cmoe / -ncmoe / -ncffn
    # 和 draft 版；**没有**「改专家个数」那种开关，个数是模型权重里固化的）。
    # 显存不够时装更大的 MoE 用它，代价是慢。0 / 没填 = 全放显存。
    # 两道闸：dense 模型不给（它没有专家张量，塞进去是无效参数），
    # 模型不是 MoE 时页面也不会显示这个框（见 is_moe / openEdit）。
    ncmoe = int(m.get("ncmoe") or 0)
    if ncmoe > 0 and m.get("moe"):
        cmd += ["-ncmoe", str(ncmoe)]

    cmd += [
        "--cache-type-k", m.get("kv", "q4_0"),
        "--cache-type-v", m.get("kv", "q4_0"),
        "-ngl", "999", "-fa", "on", "--jinja",
        "-c", str(m.get("ctx", 32768)),
        "-a", m["id"],                      # 让 /v1/models 里显示我们的 id
        "--host", "127.0.0.1",              # 只给管理器连，外面走 8091 代理
        "--port", str(ENGINE_PORT),
        "--api-key", API_KEY,
    ]
    cmd += list(m.get("extra", []))
    return cmd


# ────────────────────────── 运行时（生命周期） ──────────────────────────

class Runtime:
    """管理 llama-server 的启停。所有状态变更都在 self.lock 下做。

    切换流程（必须严格按这个顺序，否则显存不够）：
        kill 旧进程 -> 轮询等显存回落到 VRAM_IDLE_MB 以下 -> 启动新进程 -> 等 /health
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.current_id: str | None = None
        self.phase = "idle"          # idle | stopping | loading | ready | error
        self.message = "未加载"
        self.started_at: float | None = None
        self.load_seconds: float | None = None
        self.last_error = ""
        self.paused = False
        self.load_log: deque[str] = deque(maxlen=400)
        self._logfh = None
        self.models: dict = {}
        self._ext_cache: tuple[float, list[int]] = (0.0, [])
        # 自己刚杀掉的 PID -> 杀掉的时刻。taskkill 返回后它还会在 tasklist 里
        # 停留一小会儿，这期间不能把它算成「外部进程」（见 external_pids）。
        self._own_dead: dict[int, float] = {}
        self.last_tps: float | None = None      # 上一轮输出速度 (t/s)
        self.last_accept: int | None = None     # 上一轮投机解码接受率 (%)
        self.last_tps_at: float | None = None
        # 每次推理完记一条，页面上可以翻历史。跨模型**不清零** —— 每条都带着
        # 模型 id，换模型前后谁快谁慢能直接对比（last_tps 那个还是会清）。
        self.perf: deque[dict] = deque(load_perf_history(), maxlen=50)
        self.latest_perf = load_latest_perf()
        # 「不同配置下的 t/s」那段文本。惰性生成 + 缓存 —— tick 每 1.5 秒来取一次，
        # 每次都重算 + 重写文件没必要。note_tps 里会作废并立刻重写一份。
        self._tps_summary: str | None = None
        self.usage_lock = threading.Lock()
        self.model_usage = load_model_usage()

    def note_tps(self, tps: float, accept: int | None = None,
                 tokens: int | None = None, prompt_tokens: int | None = None,
                 ms: float | None = None) -> None:
        self.last_tps = round(tps, 1)
        self.last_accept = accept
        self.last_tps_at = now()
        model = self.models.get(self.current_id, {}) if self.current_id else {}
        row = {
            "tps": self.last_tps, "accept": accept, "n": tokens,
            "pin": prompt_tokens, "ms": ms,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.current_id, "name": model.get("name", self.current_id or ""),
            "ctx": model.get("ctx"), "kv": model.get("kv", "q4_0"),
            "mtp": model.get("mtp", "none"), "nmax": model.get("nmax"),
            "pmin": model.get("pmin", 0.0),
        }
        self.perf.append(row)
        if self.current_id:
            with PERF_LOCK:
                self.latest_perf[self.current_id] = row
        if self.current_id:
            with self.usage_lock:
                self.model_usage[self.current_id] = self.model_usage.get(self.current_id, 0) + 1
                row["usage"] = self.model_usage[self.current_id]
                try:
                    save_model_usage(self.model_usage)
                except OSError as e:
                    log(f"模型使用次数写入失败: {e}")
        try:
            append_perf_history(row)
        except OSError as e:
            log(f"推理速度记录写入失败: {e}")
        # 摘要**每次推理都重写**，不等谁来读：用户要的就是「程序自动写出结果」——
        # 盘上那份永远是最新的，AI 直接读 logs\tps-summary.md，不必碰界面。
        try:
            self._tps_summary = write_tps_summary(list(self.perf))
        except Exception as e:
            log(f"t/s 摘要生成失败: {e}")

    def latest_perf_snapshot(self) -> dict[str, dict]:
        with PERF_LOCK:
            return dict(self.latest_perf)

    def tps_summary_text(self) -> str:
        """页面末段「不同配置下的 t/s」要的那段文本。

        惰性生成 + 缓存：tick 每 1.5 秒来取一次，没必要每次重算重写。
        **先写到盘上再返回**，所以页面显示的、和 logs\\tps-summary.md 里躺着的是同一份文本 ——
        一份数据两个地方读，不会对不上（用户明确要求「程序自动写出结果，不要 AI 手写」）。
        """
        if self._tps_summary is None:
            # 兜住**所有**异常：这段文本是页面的附属品，它算炸了不能让 /api/state 跟着 500 ——
            # 那整个页面都会白屏（状态卡、模型列表全靠这个接口）
            try:
                self._tps_summary = write_tps_summary(list(self.perf))
            except Exception as e:
                log(f"t/s 摘要生成失败: {e}")
                self._tps_summary = "（t/s 摘要生成失败 —— 看 logs\\manager.log）"
        return self._tps_summary

    # ---- 状态快照 ----
    def snapshot(self) -> dict:
        # 切换过程会持锁几十秒。这里只「尝试」拿 0.2 秒，拿不到就照读不误 ——
        # 宁可读到中间状态，也不能让网页每 1.5 秒一次的轮询堵死（堵了就看不到进度）。
        got = self.lock.acquire(timeout=0.2)
        try:
            cur = self.models.get(self.current_id) if self.current_id else None
            mem_used, mem_total = mem_used_mb()
            with self.usage_lock:
                model_usage = dict(self.model_usage)
            return {
                "phase": self.phase,
                "current": self.current_id,
                "current_name": cur["name"] if cur else None,
                "current_note": cur.get("note", "") if cur else "",
                "message": self.message,
                "started_at": self.started_at,
                "elapsed": round(now() - self.started_at) if self.started_at else 0,
                "load_seconds": self.load_seconds,
                "last_error": self.last_error,
                "paused": self.paused,
                "pid": self.proc.pid if self.proc else None,
                "ui_version": load_ui()[1],
                "proxy_ok": PROXY_OK,
                "external": self.external_pids(),
                "vram_used_mb": vram_used_mb(),
                "vram_total_mb": GPU.latest.get("vram_limit") or VRAM_TOTAL_MB,
                "mem_used_mb": mem_used,
                "mem_total_mb": mem_total,
                "last_tps": self.last_tps,
                "last_accept": self.last_accept,
                "perf": list(self.perf),
                # 「不同配置下的 t/s」那段文本，管理器自己算的（见 tps_summary_text）
                "tps_summary": self.tps_summary_text(),
                "latest_perf": self.latest_perf_snapshot(),
                "model_usage": model_usage,
                "gpu": GPU.snapshot(),
                # 主力排最前，然后按作者聚在一起，同作者的按名字排
                "models": sorted(
                    self.models.values(),
                    key=lambda m: (not m.get("star"), not m.get("verified"),
                                   (m.get("author") or "~").lower(), m.get("name", "")),
                ),
                "logs": list(self.load_log)[-60:],
            }
        finally:
            if got:
                self.lock.release()

    def _say(self, phase: str, msg: str) -> None:
        self.phase = phase
        self.message = msg
        self.load_log.append(time.strftime("%H:%M:%S ") + msg)
        log(f"[{phase}] {msg}")

    # ---- 卸载 ----
    def unload(self, reason: str = "手动卸载") -> bool:
        with self.lock:
            self._unload_locked(reason)
            # _unload_locked 只管停进程、等显存回落，phase 的归属由调用方决定。
            if self.phase == "stopping":
                self._say("idle", f"未加载（{reason}）")
            # 调用方必须用这个结果闸住后续加载；旧进程没退出时不能再启一个。
            return self.proc is None and self.current_id is None and self.phase == "idle"

    def _unload_locked(self, reason: str) -> None:
        if self.proc is None and self.current_id is None:
            self._say("idle", "当前没有加载模型")
            return
        self._say("stopping", f"正在停止 {self.current_id or '进程'}（{reason}）")
        try:
            self._kill_proc()
        except Exception as e:
            self.last_error = f"卸载失败，进程仍可能占用显存：{e}"
            self._say("error", self.last_error)
            return
        self.current_id = None
        self.started_at = None
        ok = self._wait_vram_free()
        if not ok:
            self.load_log.append("⚠ 显存回落超时，仍继续（可能要小心溢出）")

    def external_pids(self) -> list[int]:
        """不是本管理器启动的 llama-server。

        这些是用户手动双击 bat 跑起来的（或者别的工具），正在聊天用着。
        管理器不会去杀它们，只会拒绝加载并让用户自己决定要不要接管。

        `_own_dead` 里是**自己刚杀掉的** PID：taskkill 返回成功之后，进程还会在
        tasklist 里停留一小会儿，而那时 self.proc 已经清空了 —— 不过滤掉的话，
        管理器会把自己刚停的引擎当成「外部进程」而拒绝加载，也就是「保存并重新加载
        必失败、隔十几秒手动再点一次才成功」（2026-09-26 实测，见 _kill_proc）。
        """
        for pid, t_killed in list(self._own_dead.items()):
            if now() - t_killed > OWN_DEAD_GRACE:
                self._own_dead.pop(pid, None)
        t, cached = self._ext_cache
        if now() - t < 3.0:                     # tasklist 有点慢，缓存 3 秒
            return [p for p in cached if p not in self._own_dead]
        mine = self.proc.pid if self.proc else None
        pids = [p for p in engine_pids() if p != mine and p not in self._own_dead]
        self._ext_cache = (now(), pids)
        return pids

    def _kill_proc(self) -> None:
        """只杀自己启动的那个进程树；taskkill 成功后不再重复长轮询。"""
        if self.proc is not None:
            proc = self.proc
            try:
                result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                        capture_output=True, text=True, timeout=15,
                                        creationflags=CREATE_NO_WINDOW)
            except Exception as e:
                raise RuntimeError(f"停止 PID {proc.pid} 失败：{e}") from e
            if result.returncode != 0:
                # 仅用端口快查确认 adopted 引擎还在；不调用 poll() 的 tasklist 慢路径。
                alive = (pid_on_port(ENGINE_PORT) == proc.pid
                         if isinstance(proc, _AdoptedProc) else proc.poll() is None)
                if alive:
                    detail = (result.stderr or result.stdout or "taskkill 返回失败").strip()
                    raise RuntimeError(f"PID {proc.pid} 未退出：{detail}")
            else:
                # taskkill 只保证「已下令终止」。原来这里是 wait(0)（等于不等），
                # 于是紧接着的 external_pids() 还能在 tasklist 里看到这个 PID，
                # 而 self.proc 已经清空 → 自己刚停的引擎被当成外部进程，加载被自己挡住。
                # 现在真等到它从 tasklist 里消失为止（_AdoptedProc.wait 也是这么实现的），
                # 超时就直接报错 —— 宁可让用户重试，不能双开。
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 没退干净：self.proc 故意留着（跟上面 returncode!=0 那条一样），
                    # 调用方据此中止后续加载 —— 宁可多等，不能双开。
                    raise RuntimeError(f"PID {proc.pid} 5 秒内未退出（显存可能仍被占用）")
            # 记一笔：taskkill 之后它还可能在 tasklist 里停留一小会儿，
            # 这段时间别把它当成「外部进程」。
            self._own_dead[proc.pid] = now()
            self._ext_cache = (0.0, [])
            self.proc = None
        if self._logfh:
            try:
                self._logfh.close()
            except Exception:
                pass
            self._logfh = None

    def _wait_vram_free(self) -> bool:
        """轮询到显存回落为止。这一步不能省 —— 不等就会溢出到系统内存。"""
        t0 = now()
        while now() - t0 < VRAM_TIMEOUT:
            used = vram_used_mb()
            if used < 0:
                time.sleep(1.0)
                continue
            if used <= VRAM_IDLE_MB:
                self.load_log.append(f"✓ 显存已释放（{used} MB），耗时 {now()-t0:.1f}s")
                return True
            if int(now() - t0) % 5 == 0:
                self.load_log.append(f"  等待显存释放… 当前 {used} MB")
            time.sleep(0.6)
        return False

    # ---- 加载 ----
    def ensure(self, model_id: str, reason: str = "") -> tuple[bool, str]:
        """保证 model_id 处于就绪状态；已经是它就秒返回，否则切过去。"""
        with self.lock:
            if self.phase == "ready" and self.current_id == model_id and self._alive():
                return True, ""
            if model_id not in self.models:
                err = f"没有这个模型: {model_id}"
                self.last_error = err
                self._say("error", err)
                return False, err
            if self.current_id != model_id:
                self._unload_locked(f"切换到 {model_id}" + (f" · {reason}" if reason else ""))
                # taskkill 被 Windows 拒绝时，_unload_locked 保留 current_id/proc。
                # 继续 _load_locked 会制造第二个 llama-server 并挤爆显存。
                if self.proc is not None or self.current_id is not None:
                    err = self.last_error or "旧模型进程仍在运行，已取消新模型加载"
                    return False, err
            return self._load_locked(model_id, reason)

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def adopt_existing(self) -> bool:
        """认领 8092 上已经在跑的引擎。

        目的：**重启管理器时不把已加载的模型踹掉** —— 改完配置重启，模型还在，
        不用等它重新加载十秒。

        只认「本管理器风格」的引擎：命令行里是 llama-server、带 `-a <模型id>`、
        而且这个 id 在模型表里找得到。别人手动跑的引擎不带这个别名，认不出来就不碰。
        """
        if not port_busy(ENGINE_PORT):
            return False
        pid = pid_on_port(ENGINE_PORT)
        if not pid:
            return False

        # 不读命令行（wmic 没了），直接问引擎自报的别名
        mid = engine_alias()
        if mid not in self.models:
            log(f"[!] {ENGINE_PORT} 上的东西（PID {pid}）报不出可识别的别名"
                f"（{mid!r}），不认领 —— 多半是别人手动跑的引擎")
            return False

        if not engine_healthy():
            log(f"[?] {ENGINE_PORT} 上的引擎活着但 /health 不通，不认领")
            return False

        self.proc = _AdoptedProc(pid)
        self.current_id = mid
        self.phase = "ready"
        self.started_at = now()
        self.load_seconds = None
        self.message = f"{self.models[mid]['name']}（重启前就在跑，已认领）"
        self.load_log.append(f"认领了已在运行的引擎：{mid}（PID {pid}）")
        log(f"认领已有引擎：{mid}（PID {pid}）— 模型保住了，没被卸载")
        return True

    def _load_locked(self, model_id: str, reason: str = "") -> tuple[bool, str]:
        m = self.models[model_id]
        f = Path(m["file"])
        if not f.exists():
            err = f"模型文件不存在: {m['file']}"
            self.last_error = err
            self._say("error", err)
            return False, err

        if not ENGINE.exists():
            err = f"引擎不存在: {ENGINE}"
            self.last_error = err
            self._say("error", err)
            return False, err

        # 有别人手动跑着的模型就先别动 —— 显存只够一个，硬上会溢出到系统内存
        ext = self.external_pids()
        if ext:
            err = (f"检测到 {len(ext)} 个不是本管理器启动的 llama-server"
                   f"（PID {', '.join(map(str, ext))}），显存已被占满。"
                   f"先把它关掉，或者在网页上点「接管」。")
            self.last_error = err
            self._say("error", err)
            return False, err

        head = f"正在载入 {m['name']}" + (f"（{reason}）" if reason else "")
        self._say("loading", head)
        self.load_seconds = None
        self.last_error = ""
        self.last_tps = None            # 换了模型，上一轮的速度就没意义了
        self.last_accept = None         # （perf 那份历史故意不清，见 note_tps）

        self._wait_vram_free()

        LOGDIR.mkdir(parents=True, exist_ok=True)
        logpath = LOGDIR / f"engine-{model_id}.log"
        try:
            self._logfh = open(logpath, "wb")
        except Exception:
            self._logfh = subprocess.DEVNULL

        cmd = build_cmd(m)
        self.load_log.append("$ " + " ".join(cmd))
        log("启动: " + " ".join(cmd))

        t0 = now()
        self.started_at = t0
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=self._logfh, stderr=subprocess.STDOUT,
                cwd=str(ENGINE.parent), creationflags=CREATE_NO_WINDOW,
            )
        except Exception as e:
            err = f"启动失败: {e}"
            self.last_error = err
            self._say("error", err)
            return False, err

        ok, err = self._wait_ready(model_id, t0)
        if ok:
            self.load_seconds = round(now() - t0, 1)
            self.current_id = model_id
            self._say("ready", f"{m['name']} 已就绪（{self.load_seconds}s）")
            return True, ""

        self.last_error = err
        self._say("error", err)
        try:
            self._kill_proc()
        except Exception as e:
            # 已经在失败路径上了，这里再抛就会把真正的失败原因盖掉（调用方只看到
            # 一句清理报错）。留 self.proc 原地不动：下次加载会被 external_pids 挡住，
            # 不会双开。
            log(f"[!] 清理未就绪的引擎失败（{model_id}）：{e}")
        self.current_id = None
        self.started_at = None
        return False, err

    def _wait_ready(self, model_id: str, t0: float) -> tuple[bool, str]:
        """轮询 /health，直到就绪 / 进程挂掉 / 超时"""
        last_note = 0
        while now() - t0 < LOAD_TIMEOUT:
            if self.proc is None or self.proc.poll() is not None:
                return False, "引擎进程已退出：" + self._tail_log(model_id, 3)
            try:
                conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=3)
                conn.request("GET", "/health",
                             headers={"Authorization": f"Bearer {API_KEY}"})
                r = conn.getresponse()
                body = r.read()
                conn.close()
                if r.status == 200 and b"ok" in body.lower():
                    return True, ""
            except Exception:
                pass

            el = int(now() - t0)
            if el and el % 5 == 0 and el != last_note:
                last_note = el
                self.load_log.append(f"  载入中… 已 {el}s（{vram_used_mb()} MB 显存）")
            time.sleep(1.0)
        return False, f"载入超时（{LOAD_TIMEOUT}s）：" + self._tail_log(model_id, 3)

    def _tail_log(self, model_id: str, n: int) -> str:
        try:
            p = LOGDIR / f"engine-{model_id}.log"
            lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            tail = [l for l in lines[-40:] if l.strip()]
            return " | ".join(tail[-n:])[:600]
        except Exception:
            return "（日志不可读）"

    # ---- JIT：把请求里的 model 字段映射成模型 id ----
    def resolve(self, requested: str) -> str | None:
        if not requested:
            return self.current_id or DEFAULT_ID
        r = requested.strip().lower()
        if r in self.models:
            return r
        # /v1/models 里给的是「作者/模型名」，按这个反查
        if "/" in r:
            au, _, nm = r.partition("/")
            au, nm = au.strip(), nm.strip()
            for mid, m in self.models.items():
                if (m.get("author") or "").lower() == au and \
                        (m.get("name") or "").lower() == nm:
                    return mid
        for mid, m in self.models.items():
            nm = (m.get("name") or "").lower()
            stem = Path(m["file"]).stem.lower()
            if r in (nm, stem) or r == Path(m["file"]).name.lower():
                return mid
        # 模糊匹配，取最短的那个（片段越短说明越贴近）
        hits = [mid for mid, m in self.models.items()
                if r in Path(m["file"]).stem.lower() or r in (m.get("name") or "").lower()]
        if hits:
            return sorted(hits, key=len)[0]
        return None


RT = Runtime()


# GPU 温度就用 nvidia-smi 的核心温度。
# 曾经想从 HWiNFO 共享内存拿热点温度（nvidia-smi 不暴露这个传感器），
# 但 HWiNFO 免费版的共享内存限制太多（要手动开启、勾完必须重启、
# 装在 Program Files 还存不上配置、免费版 12 分钟自动关），
# 2026-09-13 按用户要求整个去掉了。

# ── 共享 GPU 内存（任务管理器 GPU 页那个「共享 GPU 内存」）──────────────
# nvidia-smi 不报这个值。它是 WDDM 层面的：显存吃紧时驱动会把一部分数据挪到
# 系统内存里，就是这里在涨 —— 平时应该是接近零，真涨起来就说明要往内存溢出了。
# 任务管理器读的是 PDH 计数器 \GPU Adapter Memory(*)\Shared Usage，我们也读同一组。
#
# 两个绕不开的点：
#   1. 用 PdhAddEnglishCounterW 而不是 PdhAddCounterW —— 中文系统上计数器名是
#      本地化的（「共享内存使用量」），带 English 的那个 API 专治这个。
#   2. 走 PDH 而不是 PowerShell / typeperf：后者每次要起进程（几百 ms），
#      1 秒采一次扛不住。PDH 读的是共享内存，几乎零开销。
#
# 坑（踩过，直接段错误）：PdhGetFormattedCounterArrayW 最后一个参数在 argtypes 里
# 必须声明成 c_void_p。写成 POINTER(POINTER(_PdhItem)) 再传个 POINTER(_PdhItem) 进去，
# ctypes 按二级指针解释 → 指错地方 → 当场 SIGSEGV。
# 还有 restype 要用 c_ulong（Windows 上是 32 位无符号）：用 c_long 的话
# PDH_MORE_DATA 会变成负数，跟 0x800007D2 比永远不相等。

_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_DOUBLE = 0x00000200
_SHARED_PATH = r"\GPU Adapter Memory(*)\Shared Usage"
_DEDICATED_PATH = r"\GPU Adapter Memory(*)\Dedicated Usage"


class _PdhUnion(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong),
                ("AnsiStringValue", ctypes.c_char_p),
                ("WideStringValue", ctypes.c_wchar_p)]


class _PdhFmtValue(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("CStatus", ctypes.c_ulong), ("u", _PdhUnion)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtValue)]


class SharedVram:
    """读「共享 GPU 内存」，单位 MB。读不到一律返回 None。

    这个值只是锦上添花，任何一步失败都不能把 GPU 监控线程带崩。
    """

    def __init__(self):
        self.pdh = None
        self.q = None
        self.h_shared = None
        self.h_ded = None
        self._open()

    def _open(self) -> None:
        try:
            p = self.pdh = ctypes.WinDLL("pdh.dll")
            p.PdhOpenQueryW.restype = ctypes.c_ulong
            p.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_void_p)]
            p.PdhAddEnglishCounterW.restype = ctypes.c_ulong
            p.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                                ctypes.c_size_t,
                                                ctypes.POINTER(ctypes.c_void_p)]
            p.PdhCollectQueryData.restype = ctypes.c_ulong
            p.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            p.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong
            p.PdhGetFormattedCounterArrayW.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]

            q = ctypes.c_void_p()
            if p.PdhOpenQueryW(None, 0, ctypes.byref(q)) != 0:
                log("[!] 共享显存：PdhOpenQuery 失败，这个值不显示了")
                return
            for attr, path in (("h_shared", _SHARED_PATH), ("h_ded", _DEDICATED_PATH)):
                h = ctypes.c_void_p()
                if p.PdhAddEnglishCounterW(q, path, 0, ctypes.byref(h)) != 0:
                    log(f"[!] 共享显存：加不了计数器 {path}")
                    return
                setattr(self, attr, h)
            p.PdhCollectQueryData(q)          # 先采一次，之后才有数可读
            self.q = q
        except Exception as e:
            log(f"[!] 共享显存初始化失败（{e}），这个值不显示了")
            self.q = None

    def _array(self, h) -> dict:
        """一个通配符计数器的所有实例 {实例名: 值}"""
        size, cnt = ctypes.c_ulong(0), ctypes.c_ulong(0)
        if self.pdh.PdhGetFormattedCounterArrayW(
                h, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(cnt),
                None) != _PDH_MORE_DATA or not size.value:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(
                h, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(cnt), buf) != 0:
            return {}
        it = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
        return {it[i].szName: it[i].FmtValue.doubleValue for i in range(cnt.value)}

    def read_mb(self) -> int | None:
        if self.q is None:
            return None
        try:
            if self.pdh.PdhCollectQueryData(self.q) != 0:
                return None
            ded = self._array(self.h_ded)
            shared = self._array(self.h_shared)
            if not ded or not shared:
                return None
            # 这组计数器每条 GPU 一个实例（独显、核显、基本显示适配器会各算一个）。
            # 挑「专用显存占用最大」的那个 = 真在干活的那块卡，不用去翻 LUID 认显卡。
            inst = max(ded, key=ded.get)
            return int(shared.get(inst, 0) / (1024 * 1024))
        except Exception:
            return None


class GpuMonitor(threading.Thread):
    """后台每 1 秒采一次 GPU 数据，给页面画四条曲线。

    为什么不放进 snapshot()：nvidia-smi 一次要 ~70ms，页面每 1.5 秒轮询一次
    就白白跑一遍。单独线程采样 + 缓存，页面直接读缓存，零开销。

    六个字段一起查和只查一个字段耗时几乎一样（都是 0.07s），所以顺手多拿几个。

    1 秒一次（2026-09-13 用户要求，原来是 3 秒）：等于常驻吃掉约 5% 的一个核，
    换来 480 秒（8 分钟）的窗口。再密就要考虑换 NVML 了（nvidia-smi 每次都要
    起一个进程，70ms 里大半是进程启动开销）。
    """

    def __init__(self, interval: float = 1.0, history: int = 480):
        super().__init__(daemon=True, name="gpu-monitor")
        self.interval = interval
        self.hist: deque[dict] = deque(maxlen=history)    # 每次采样 {vram, mem, power, temp}
        self.latest: dict = {}
        self.temp_max: int | None = None      # 开机跑到现在见过的最高核心温度
        self.power_max: float | None = None   # 同上，功耗

    @staticmethod
    def _num(x: str, cast=float):
        try:
            return cast(x)
        except Exception:
            return None          # 有些卡不支持某些字段，返回 [N/A]

    def sample(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu="
                 "power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
                creationflags=CREATE_NO_WINDOW)
            parts = [x.strip() for x in out.stdout.strip().split(",")]
            if len(parts) < 6:
                return
            power = self._num(parts[0])
            pw = round(power, 1) if power is not None else None
            vram = self._num(parts[3], int)
            mem_used, mem_total = mem_used_mb()
            temp = self._num(parts[2], int)
            self.latest = {
                "power": pw,
                "util": self._num(parts[1], int),
                "temp": temp,
                "vram": vram,
                "vram_limit": self._num(parts[4], int),
                "limit": self._num(parts[5]),
                "mem": mem_used if mem_used > 0 else None,
                "mem_limit": mem_total if mem_total > 0 else None,
                "shared": SHARED_VRAM.read_mb(),      # 共享 GPU 内存（MB）
            }
            if temp is not None and (self.temp_max is None or temp > self.temp_max):
                self.temp_max = temp
            if pw is not None and (self.power_max is None or pw > self.power_max):
                self.power_max = pw
            self.hist.append({"vram": vram, "mem": mem_used if mem_used > 0 else None,
                              "power": pw, "temp": temp})
        except Exception:
            pass

    def run(self) -> None:
        self.sample()                      # 先采一次，别让页面开屏空着
        while True:
            time.sleep(self.interval)
            self.sample()

    def snapshot(self) -> dict:
        d = dict(self.latest)
        h = list(self.hist)
        # 四条曲线拆成四个数组，前端好画；比 [{"vram":..},..] 省一半字节
        d["vram_hist"] = [x.get("vram") for x in h]
        d["mem_hist"] = [x.get("mem") for x in h]
        d["power_hist"] = [x.get("power") for x in h]
        d["temp_hist"] = [x.get("temp") for x in h]
        # 最高温 / 最高功耗都是累计值、不在 latest 里，单独塞进去
        d["temp_max"] = self.temp_max
        d["power_max"] = self.power_max
        return d


SHARED_VRAM = SharedVram()      # 共享 GPU 内存，由 GpuMonitor 每秒读一次
GPU = GpuMonitor()


# ────────────────────────── 速度指标（从 llama.cpp 的 timings 里抠） ──────────────────────────

_RE_TPS = re.compile(rb'"predicted_per_second"\s*:\s*([0-9.]+)')
_RE_PRED_N = re.compile(rb'"predicted_n"\s*:\s*(\d+)')
_RE_DRAFT_N = re.compile(rb'"draft_n"\s*:\s*(\d+)')
_RE_DRAFT_OK = re.compile(rb'"draft_n_accepted"\s*:\s*(\d+)')
# 输入 token 总数 = prompt_n（这次真算的）+ cache_n（命中前缀缓存、没重算的）。
# **两个都要**：只取 prompt_n 的话，连续对话里命中缓存的那一大截会凭空消失，
# 显示出来的「输入」会远小于实际上下文。字段名照 llama.cpp 的 timings 抄，
# 2026-09-21 拿本机 8092 的真实响应核对过（prompt_n=25 / cache_n=0 / predicted_ms=436.606）。
_RE_PROMPT_N = re.compile(rb'"prompt_n"\s*:\s*(\d+)')
_RE_CACHE_N = re.compile(rb'"cache_n"\s*:\s*(\d+)')
# 输出总耗时，毫秒。别拿 n/tps 反推 —— 那算出来的是纯解码时间，
# predicted_ms 是 llama.cpp 自己记的整段生成墙钟，两者实测差个零点几秒。
_RE_PRED_MS = re.compile(rb'"predicted_ms"\s*:\s*([0-9.]+)')


def note_timings(blob: bytes) -> None:
    """从 llama.cpp 的响应里提取生成速度。

    llama-server 自己就在算这个，不用我们拿秒表估：
      非流式 —— 顶层有 timings 字段，整个 body 里就能搜到
      流式   —— timings 在最后一个数据块里，所以调用方传尾部缓冲进来

    同时抠出 predicted_n（这次生成了几个 token）**一起记下来**。
    必须记，不然数字会骗人：首 token 有个固定开销，只吐 8 个 token 的一问一答
    平均下来能显示成 6 t/s（实测），正常长回复才是 108 t/s —— 差 17 倍。
    有了 token 数，看历史时一眼就知道那个洼地是「回复太短」还是真掉速了。

    2026-09-21 又补了两样（用户要求）：**输入 token 总数**（prompt_n + cache_n）
    和**输出总耗时**（predicted_ms）。前者解释「为什么这次慢」—— 上下文越长首 token 越贵；
    后者是用户真正关心的那个数（t/s 是平均值，看不出这次一共等了多久）。
    """
    try:
        m = _RE_TPS.search(blob)
        if not m:
            return
        accept = None
        dn, da = _RE_DRAFT_N.search(blob), _RE_DRAFT_OK.search(blob)
        if dn and da and int(dn.group(1)) > 0:
            accept = round(int(da.group(1)) * 100 / int(dn.group(1)))
        pn = _RE_PRED_N.search(blob)
        ip, ic = _RE_PROMPT_N.search(blob), _RE_CACHE_N.search(blob)
        pin = (int(ip.group(1)) + (int(ic.group(1)) if ic else 0)) if ip else None
        ms = _RE_PRED_MS.search(blob)
        RT.note_tps(float(m.group(1)), accept, int(pn.group(1)) if pn else None,
                    pin, round(float(ms.group(1)), 1) if ms else None)
    except Exception:
        pass


# ────────────────────────── 系统提示词注入 ──────────────────────────

def inject_system(raw: bytes, mid: str | None) -> bytes:
    """把模型配置里的 system_prompt 塞到请求最前面。

    为什么做在代理层：TAVO 那边一个字都不用改，换模型时提示词跟着模型走，
    也不会污染 chat_template.jinja（那个模板所有 Qwen 模型共用）。

    HauhauCS 这类微调版时不时蹦英文，靠这一段压住。
    """
    if not mid or mid not in RT.models:
        return raw
    sp = (RT.models[mid].get("system_prompt") or "").strip()
    if not sp:
        return raw
    try:
        req = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return raw                     # /v1/completions 之类没有 messages，原样放行
    try:
        if msgs[0].get("role") == "system":
            cur = msgs[0].get("content") or ""
            if sp not in cur:          # 已经注入过就别叠加（同一请求重试时）
                msgs[0]["content"] = f"{sp}\n\n{cur}" if cur else sp
        else:
            msgs.insert(0, {"role": "system", "content": sp})
        return json.dumps(req, ensure_ascii=False).encode("utf-8")
    except Exception:
        return raw


# ────────────────────────── 参数编辑（手机上改） ──────────────────────────

# 允许改的字段。改动会写进 models.json，所以能持久保存
CONFIG_FIELDS = ("ctx", "nmax", "pmin", "kv", "mtp", "template", "extra",
                 "author", "name", "note", "speed", "star", "system_prompt", "mmproj",
                 "ncmoe")


def save_config(mid: str, patch: dict) -> tuple[bool, str]:
    with RT.lock:
        if mid not in RT.models:
            return False, f"没有这个模型: {mid}"
        patch = {k: v for k, v in patch.items() if k in CONFIG_FIELDS and v is not None}
        if isinstance(patch.get("extra"), str):
            patch["extra"] = patch["extra"].split()
        if not patch:
            return False, "没有要改的字段"

        doc: dict = {}
        if REGISTRY.exists():
            try:
                doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
            except Exception as e:
                return False, f"models.json 解析失败: {e}"
        doc.setdefault("models", {})
        entry = dict(doc["models"].get(mid) or {})
        entry.update(patch)
        doc["models"][mid] = entry
        try:
            REGISTRY.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        except Exception as e:
            return False, f"写 models.json 失败: {e}"

        RT.models[mid].update(patch)
        if any(k != "star" for k in patch):
            RT.models[mid]["verified"] = True
        log(f"改参数 {mid}: {patch}")
        return True, ""


def reload_model(mid: str, reason: str = "参数改了，重载") -> None:
    """设置已保存后卸载，再每秒检查引擎端口与显存，空闲后自动加载。"""
    # 同一把 RLock 覆盖卸载 + 加载，避免另一个请求在两步之间抢先启模。
    with RT.lock:
        if not RT.unload(reason):
            msg = RT.last_error or "旧模型进程仍在运行"
            log(f"重载已取消，避免双模型：{msg}")
            return
        started = now()
        last_wait = 0
        while now() - started < 120:
            # Windows 可能有手动启动的引擎/提升权限进程；每秒刷新检测，绝不替用户杀它。
            RT._ext_cache = (0.0, [])
            external = RT.external_pids()
            port_pid = pid_on_port(ENGINE_PORT)
            used = vram_used_mb()
            port_free = port_pid is None and not port_busy(ENGINE_PORT)
            memory_free = used < 0 or used <= VRAM_IDLE_MB
            if not external and port_free and memory_free:
                break
            if int(now() - started) != last_wait:
                last_wait = int(now() - started)
                log(f"等待引擎卸载完成… {last_wait}s，外部进程={external or '无'}，"
                    f"端口PID={port_pid or '空闲'}，显存={used if used >= 0 else '未知'}MB")
            time.sleep(1.0)
        else:
            msg = "等待 8092 端口/显存释放超时（120 秒），未启动新实例"
            RT.last_error = msg
            RT._say("error", msg)
            return
        ok, err = RT.ensure(mid, reason)
        if not ok:
            log(f"新模型加载失败：{err}")


# ────────────────────────── HTTP 公共部分 ──────────────────────────

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}


class Base(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LLMManager/1.0"

    def log_message(self, fmt, *args):        # 静音默认的 stderr 日志
        pass

    # ---- 响应助手 ----
    def send_json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")   # 手机浏览器别缓存状态
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: int = 200, ctype="text/html; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def client_key(self) -> str:
        a = self.headers.get("Authorization") or ""
        if a.lower().startswith("bearer "):
            return a[7:].strip()
        q = parse_qs(urlparse(self.path).query)
        return (q.get("k") or [""])[0]

    def authed(self) -> bool:
        return self.client_key() == API_KEY

    def deny(self):
        self.send_json({"error": {"message": "API key 不对", "type": "invalid_request_error"}}, 401)


# ────────────────────────── 控制台和 REST API ──────────────────────────

class ConsoleHandler(Base):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html, _ = load_ui()
            self.send_text(html)
        elif path == "/api/state":
            if not self.authed():
                return self.deny()
            self.send_json(RT.snapshot())
        elif path == "/api/logs":
            if not self.authed():
                return self.deny()
            self.send_json({"logs": list(RT.load_log),
                            "manager": tail_manager_log()})
        elif path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if not self.authed():
            return self.deny()
        body = self.read_json()

        if path == "/api/load":
            mid = body.get("id") or body.get("model")
            if not mid:
                return self.send_json({"ok": False, "error": "缺少 id"}, 400)
            if mid not in RT.models:
                return self.send_json({"ok": False, "error": f"没有这个模型: {mid}"}, 404)
            threading.Thread(target=RT.ensure, args=(mid, "手机网页"),
                             daemon=True).start()
            # 不在这里返回 state —— 加载线程可能已经抢到锁了，snapshot 会一直等到
            # 模型载完才返回，点按钮的手机会以为卡死了。状态让前端自己轮询。
            self.send_json({"ok": True, "message": f"开始加载 {mid}"})

        elif path == "/api/unload":
            threading.Thread(target=RT.unload, args=("手机网页",), daemon=True).start()
            self.send_json({"ok": True, "message": "开始卸载"})

        elif path == "/api/pause":
            # 同样不抢锁：加载期间点暂停也得立刻生效
            RT.paused = bool(body.get("paused", True))
            RT.message = "已暂停（不响应请求、不自动切换）" if RT.paused else "已恢复"
            log(f"暂停状态 = {RT.paused}")
            self.send_json({"ok": True, "paused": RT.paused})

        elif path == "/api/refresh":
            RT.models = discover_models()
            self.send_json({"ok": True, "count": len(RT.models)})

        elif path == "/api/takeover":
            threading.Thread(target=takeover, args=("手机网页",), daemon=True).start()
            self.send_json({"ok": True, "message": "开始接管，会杀掉手动启动的模型"})

        elif path == "/api/sleep":
            # 顺序不能反：先把响应写回去，再起线程去睡。
            # 反过来的话手机那边连接先断、只看到「请求失败」，会以为没睡成又点一次。
            self.send_json({"ok": True, "message": "正在进入睡眠…"})
            threading.Thread(target=sleep_pc, args=("手机网页",), daemon=True).start()

        elif path == "/api/shutdown":
            # 备选路径：真关机之后要有人输 PIN 管理器才会回来，远程用请走 /api/sleep。
            #
            # **两步走**（2026-09-21 用户定的）：不带 force 的这一发**只探测不关机** ——
            # 有程序注册了关机阻断就把名字报回去，让手机先看清楚是谁挡着，再决定要不要强制。
            # 没有阻断者才直接关。理由见 shutdown_blockers() 上面那段。
            force = bool(body.get("force"))
            if not force:
                blocked = shutdown_blockers()
                if blocked:
                    return self.send_json({"ok": False, "blocked": True,
                                           "blockers": blocked})
            self.send_json({"ok": True, "message": "正在关机…"})
            threading.Thread(target=shutdown_pc, args=("手机网页", force),
                             daemon=True).start()

        elif path == "/api/restart":
            # 改了 config.json / 源码之后用它 —— 省得让人跑到电脑前杀进程再双击 start.bat。
            #
            # **闸门**：正在生成、或正在切换模型时不给重启。点这个按钮的意图通常是
            # 「让新配置生效」，不是「砍掉我正在等的回复」；真忙就让他几秒后再点。
            # 睡眠/关机不设这道闸门，因为那两个本来就是「我要断服务」。
            if RT.phase in ("loading", "stopping"):
                return self.send_json(
                    {"ok": False,
                     "message": "正在切换模型，等它结束再点重启。"}, 409)
            if engine_generating():
                return self.send_json(
                    {"ok": False,
                     "message": "引擎正在生成回复，重启会打断它。等这条出完再点。"}, 409)
            ok, err = spawn_restart()
            if not ok:
                return self.send_json({"ok": False, "message": err}, 500)
            self.send_json({"ok": True, "message": "正在重启，几秒后自动恢复…"})
            exit_soon()          # 响应已经发出去了，再由定时器把本进程收掉

        elif path == "/api/config":
            mid = body.get("id")
            if not mid:
                return self.send_json({"ok": False, "error": "缺少 id"}, 400)
            ok, err = save_config(mid, body.get("patch") or {})
            if ok and body.get("reload"):
                threading.Thread(target=reload_model, args=(mid,),
                                 daemon=True).start()
            self.send_json({"ok": ok, "error": err})

        else:
            self.send_json({"error": "not found"}, 404)


# ────────────────────────── 8091：OpenAI 代理（含 JIT） ──────────────────────────

class ProxyHandler(Base):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        # 浏览器直接打开 8091 时也给它控制台页面。
        # 主入口 80 和兼容入口 8091 都提供同一个控制台；80 上的 /v1 同时提供模型 API。
        # 没带 key 时页面会显示输入框（后端这里返回 401，页面自己弹）。
        if path in ("/", "/index.html") or path.startswith("/api/"):
            return ConsoleHandler.do_GET(self)
        if not self.authed():
            return self.deny()
        if path == "/v1/models" or path == "/models":
            # 返回全部模型，这样前端下拉框里能看到所有可选模型（选了就触发 JIT）。
            # id 直接写成「作者/模型名」—— 前端下拉框一般只认 id，写成这样才看得出
            # 是哪个作者的。回传过来时 Runtime.resolve() 认得这个格式。
            data = []
            for m in RT.snapshot()["models"]:
                au = m.get("author") or ""
                label = f"{au}/{m['name']}" if au else m["name"]
                data.append({
                    "id": label,
                    "object": "model",
                    "created": 0,
                    "owned_by": au or "local",
                    "name": label,          # 有些前端认这个字段
                })
            return self.send_json({"object": "list", "data": data})
        if path == "/health":
            return self.send_json({"status": "ok", "model": RT.current_id,
                                   "phase": RT.phase})
        self.proxy("GET")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/"):
            return ConsoleHandler.do_POST(self)      # 8091 上的控制台操作
        if not self.authed():
            return self.deny()

        # 读请求体，拿 model 字段决定要不要切换
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        want = None
        try:
            want = (json.loads(raw.decode("utf-8")) or {}).get("model")
        except Exception:
            pass

        if not RT.paused:
            mid = RT.resolve(want)
            if mid is None:
                mid = RT.current_id or DEFAULT_ID
            ok, err = RT.ensure(mid, reason=f"来自请求 model={want!r}")
            if not ok:
                return self.send_json(
                    {"error": {"message": f"模型切换失败: {err}", "type": "server_error"}}, 500)
        else:
            if RT.phase != "ready":
                return self.send_json(
                    {"error": {"message": "管理器已暂停，手机上点恢复", "type": "server_error"}}, 503)
            mid = RT.current_id

        raw = inject_system(raw, mid)      # 语言要求之类的系统提示词
        self.proxy("POST", raw)

    # ---- 转发 ----
    def proxy(self, method: str, raw: bytes | None = None):
        # 转发时用正确的 key（llama-server 也校验）
        headers = {"Authorization": f"Bearer {API_KEY}"}
        for k, v in self.headers.items():
            if k.lower() in HOP_BY_HOP or k.lower() == "authorization":
                continue
            headers[k] = v
        headers["Host"] = f"127.0.0.1:{ENGINE_PORT}"

        if raw is None:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else None

        try:
            conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=1800)
            conn.request(method, self.path, body=raw, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            return self.send_json(
                {"error": {"message": f"内部引擎连不上（{e}）。当前状态：{RT.message}",
                           "type": "server_error"}}, 502)

        out = {}
        for k, v in resp.getheaders():
            if k.lower() in HOP_BY_HOP:
                continue
            out[k] = v

        clen = resp.getheader("Content-Length")
        try:
            self.send_response(resp.status)
            for k, v in out.items():
                self.send_header(k, v)
            if clen is not None:
                self.send_header("Content-Length", clen)
                self.end_headers()
                left = int(clen)
                buf = bytearray()
                while left > 0:
                    chunk = resp.read1(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if len(buf) < 2_000_000:      # 留一份读 timings 就够，别把长回答全缓存
                        buf += chunk
                    left -= len(chunk)
                note_timings(bytes(buf))
            else:
                # SSE 流式：不知道长度，用 chunked 边收边发
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                tail = b""
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                    tail = (tail + chunk)[-8192:]   # timings 在最后一个数据块里
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                note_timings(tail)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客户端自己断了（常见于切模型期间的前端超时），不用管
            pass
        except Exception as e:
            log(f"[warn] 转发中断: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self.close_connection = True


# ────────────────────────── 网页控制台 ──────────────────────────

# 网页的 HTML/CSS/JS 原本写在这里（一个 PAGE 常量），占了整个文件 30%，
# 把 model_manager.py 撑到 1974 行 / 81KB，在记事本里都打不开。
# 现在拆到同目录的 ui.html 了 —— 改界面直接编辑那个文件，刷新页面即生效。


# ────────────────────────── 启动 ──────────────────────────

class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    """等端口空出来 —— 重启时新进程用它等旧进程放手。

    为什么不等旧 PID 退出：PID 会被系统复用，而端口不会骗人；而且这里直接
    验证的正是「能不能绑」，跟上面 port_busy 问的是同一件事。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not port_busy(port):
            return True
        time.sleep(0.3)
    return False


def spawn_restart() -> tuple[bool, str]:
    """起一个脱离本进程的新管理器实例。本进程随后退出，由调用方安排。

    为什么绕一道 VBS 而不是直接 subprocess.Popen：wscript 那句
    `sh.Run(cmd, 0, False)` 第三个参数 False = **不等它结束**，新进程也不继承
    本进程的控制台 —— 本进程一退它照样活着。直接 Popen 的话子进程会跟着父进程
    一起走，重启就变成了自杀。

    新进程必须带 `--restart-wait`：它会先等 MANAGER_PORT 空出来再启动。不带的话
    它会撞上还没退干净的自己，在 main() 里看到「端口已被占用」直接退出 ——
    手机上看起来就是「点了重启，然后再也没起来」。

    解释器用 sys.executable，跟当前进程保持一致：本来就是 pythonw 起的就还是
    pythonw，手动用 python.exe 起的就还是 python.exe。
    """
    vbs = LOGDIR / "restart.vbs"          # 放 logs\ 下，不往根目录丢临时文件
    # wscript 按系统 ANSI 读 .vbs，所以写 mbcs —— 路径里有中文用户名时才不乱码。
    # 文件内容保持纯 ASCII，理由同 autostart.py 里那段。
    # 端口原样带上：用 --console-port 8080 起的管理器，重启后还得在 8080，
    # 不能默默跳回默认的 80（那会让手机上的书签失效）。
    args = (f"--console-port {MANAGER_PORT} --proxy-port {PROXY_PORT} "
            f"--engine-port {ENGINE_PORT} --restart-wait {MANAGER_PORT}")
    body = (
        "' Written by the manager's restart button. Safe to delete.\r\n"
        "Set sh = CreateObject(\"WScript.Shell\")\r\n"
        f'sh.CurrentDirectory = "{BASE}"\r\n'
        f'sh.Run """{sys.executable}"" ""{Path(__file__).resolve()}"" {args}", 0, False\r\n'
    )
    try:
        vbs.parent.mkdir(parents=True, exist_ok=True)
        vbs.write_text(body, encoding="mbcs")
    except Exception as e:
        return False, f"写不了 {vbs.name}: {e}"
    try:
        subprocess.Popen(["wscript", str(vbs)],
                         creationflags=CREATE_NO_WINDOW, close_fds=True)
    except Exception as e:
        return False, f"起不了 wscript: {e}"
    return True, ""


def exit_soon(delay: float = 1.5) -> None:
    """延迟一点再退出，**先把 HTTP 响应发出去**，否则手机上只会看到「请求失败」。

    用 os._exit 而不是 sys.exit：跳过解释器收尾（还要等 daemon 线程），端口放得
    快；而且这个管理器**故意不走任何清理** —— 引擎要留着给新实例的
    adopt_existing() 认领，不然重启就变成「卸载又重载」，白多花十几秒。
    """
    def bye():
        log("重启：本进程退出，等新实例接手")
        os._exit(0)
    threading.Timer(delay, bye).start()


def engine_generating() -> bool:
    """引擎这会儿有没有在吐 token —— 问 /slots 的 is_processing。

    问不到就当没在忙：引擎没加载时 /slots 本来就连不上，那正是该允许重启的时候；
    引擎卡死时同理（那种情况重启恰好是解法）。
    """
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=2)
        conn.request("GET", "/slots", headers={"Authorization": f"Bearer {API_KEY}"})
        data = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return any(s.get("is_processing") for s in data if isinstance(s, dict))
    except Exception:
        return False


def lan_ip() -> str:
    """挑一个局域网 IPv4。开机自启时主机名可能还没解析好，所以多试几条路。"""
    cands: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            cands.append(info[4][0])
    except Exception:
        pass
    try:                                   # 兜底：连一下外部地址看路由选中哪个网卡
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.3)
            s.connect(("8.8.8.8", 80))
            cands.append(s.getsockname()[0])
    except Exception:
        pass

    order = (lambda ip: 0 if ip.startswith("192.168.") else
                       1 if ip.startswith("10.") else
                       2 if re.match(r"172\.(1[6-9]|2\d|3[01])\.", ip) else 3)
    real = [ip for ip in dict.fromkeys(cands) if not ip.startswith("127.")]
    return min(real, key=order) if real else "127.0.0.1"


_proxy_server = None


def start_proxy() -> bool:
    """把 8091 的 OpenAI 入口拉起来。端口被占时返回 False，不抛异常。"""
    global PROXY_OK, _proxy_server
    if PROXY_OK:
        return True
    try:
        _proxy_server = Server(("0.0.0.0", PROXY_PORT), ProxyHandler)
    except OSError as e:
        log(f"[!] {PROXY_PORT} 代理起不来：{e}")
        return False
    threading.Thread(target=_proxy_server.serve_forever, daemon=True).start()
    PROXY_OK = True
    log(f"OpenAI 入口已监听 :{PROXY_PORT}")
    return True


def takeover(reason: str = "手动接管") -> tuple[bool, str]:
    """收编用户手动启动的 llama-server：腾出 8091 和显存，交给管理器管。"""
    ext = RT.external_pids()
    if ext:
        msg = f"接管：终止外部 llama-server（PID {', '.join(map(str, ext))}）"
        log(msg)
        RT.load_log.append(msg)
    else:
        RT.load_log.append("接管：没发现外部进程，只重启代理")
    RT.unload("接管")
    kill_engines()
    RT._ext_cache = (0.0, [])
    RT._wait_vram_free()
    ok = start_proxy()
    return ok, "" if ok else f"端口 {PROXY_PORT} 还是被别的程序占着"


# ── 电源：睡眠(S3) 是主路，真关机只是备选 ──
#
# **2026-09-15 把这里原来那段结论推翻了。** 旧结论是「管理器用 `shutdown /s` 关机会让下次
# 开机要 PIN，改成 ExitWindowsEx(EWX_POWEROFF) 就没事」。实测证伪 —— 两条路根本等价：
#   · 开始菜单那次 Kernel-Power 109 的 Action = 6 (PowerActionShutdownOff)，
#     管理器改用 EWX_POWEROFF 之后**也是 6**（21:40:58、21:46:46 两次都对上了）；
#   · 「开机→会话建立」开始菜单后 27 秒、EWX_POWEROFF 后 34 秒，本来就在同一档，
#     26~40 秒 vs 76/138 秒的差别只是**人当时在不在**（不在就得等人来输 PIN）；
#   · 更要命的是 Winlogon 键里**根本没有** AutoAdminLogon / DefaultUserName / DefaultPassword
#     —— 这台机器从来没配过自动登录，旧结论里「开始菜单关机后会自动登录」这个前提不存在。
#
# 所以真正的不变式是：**任何真正的关机（不管谁发起、走哪个 API）都必然停在登录界面**；
# 而管理器的自启挂在用户登录上（启动文件夹）→ 关机之后没人输 PIN，80/8091 就永远起不来。
# 结论：远程要能用，出路不是「换个姿势关机」，而是**别关机** ——
#   → 走睡眠(S3)：会话和进程原样留在内存里，唤醒后管理器还在、不需要任何人登录；
#     而且 S3 正是 WOL / WoWLAN 官方支持的状态，WiFi 就能唤醒（关机态 S5 唤不醒）。
# 代价：待机耗几瓦；GPU 跟着掉电 → 睡前必须先卸模型（见 sleep_pc）。
# 实测数据、唤醒侧要跑的管理员命令 → docs\power.md。
EWX_POWEROFF = 0x00000008
EWX_FORCE = 0x00000004
SHTDN_REASON_FLAG_PLANNED = 0x80000000


def shutdown_blockers() -> list[dict]:
    """谁在挡着关机 —— 返回 [{pid, name, reason}]，没有就返回 []。

    **2026-09-21 加这个函数的经过（关机键「坏了」的真正原因）：**
    用户报「完全关机点了没反应」。查事件日志 —— 09-21 01:49:12 和 06:11:36 两次
    都有 `pythonw.exe` 发起的 1074（关机请求），但**后面既没有 winlogon 的 1074，
    也没有 6006 / 109**，等于卡在半路；而 09-15 能关掉的那两次，这几条是齐的。

    卡在哪：Windows 收到关机请求后挨个问程序能不能关，**有程序调用过
    ShutdownBlockReasonCreate() 的话，系统会停在「这些应用阻止关机」那一页
    等人点「仍要关机」** —— 而「完全关机」这个功能存在的意义恰恰是人不在机器前。
    本机注册这条的是 Cherry Studio，原因串 "Cherry Studio is finishing background work"。
    所以关机前先探测一遍，把名字报给手机（用户 2026-09-21 选的两步走方案）。

    注意这是**查询**：ShutdownBlockReasonQuery 只对调用过 Create 的窗口返回 TRUE，
    普通窗口一律 FALSE，不会误报。
    """
    out: list[dict] = []
    try:
        # 在函数里 import：`ctypes.wintypes` 在非 Windows 上直接抛，
        # 放模块顶层会让这个文件在别的机器上连 import 都过不去。
        from ctypes import wintypes
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        cb_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u32.EnumWindows.argtypes = [cb_t, wintypes.LPARAM]
        u32.EnumWindows.restype = wintypes.BOOL
        u32.ShutdownBlockReasonQuery.argtypes = [wintypes.HWND, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)]
        u32.ShutdownBlockReasonQuery.restype = wintypes.BOOL
        u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                                 ctypes.POINTER(wintypes.DWORD)]
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                   wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD)]
        k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        def proc_name(pid: int) -> str:
            h = k32.OpenProcess(0x1000, False, pid)      # QUERY_LIMITED_INFORMATION
            if not h:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(512)
                n = wintypes.DWORD(512)
                if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                    return buf.value.rsplit("\\", 1)[-1]
            finally:
                k32.CloseHandle(h)
            return ""

        def visit(hwnd, _):
            buf = ctypes.create_unicode_buffer(512)
            n = wintypes.DWORD(512)
            if u32.ShutdownBlockReasonQuery(hwnd, buf, ctypes.byref(n)):
                pid = wintypes.DWORD()
                u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                out.append({"pid": pid.value, "name": proc_name(pid.value),
                            "reason": buf.value})
            return True

        u32.EnumWindows(cb_t(visit), 0)
    except Exception as e:
        log(f"[!] 查关机阻断者失败：{e}")
    return out


def _enable_shutdown_privilege() -> bool:
    """自己把 SE_SHUTDOWN_NAME 打开。

    `shutdown.exe` 内部替自己开了这个特权，所以走命令行时不用管；**直接调 API 就得自己来**，
    否则 ExitWindowsEx 直接失败。实测本机普通权限就开得上（AdjustTokenPrivileges → err=0），
    不需要管理员。结构体 sizeof：LUID 8 / LUID_AND_ATTRIBUTES 12 / TOKEN_PRIVILEGES 16。
    """
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", ctypes.c_ulong)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", ctypes.c_ulong),
                    ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY, SE_PRIVILEGE_ENABLED = 0x20, 0x8, 0x2

    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    advapi.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                        ctypes.POINTER(ctypes.c_void_p)]
    advapi.OpenProcessToken.restype = ctypes.c_int
    advapi.LookupPrivilegeValueW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                                             ctypes.POINTER(LUID)]
    advapi.LookupPrivilegeValueW.restype = ctypes.c_int
    advapi.AdjustTokenPrivileges.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                             ctypes.POINTER(TOKEN_PRIVILEGES),
                                             ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
    advapi.AdjustTokenPrivileges.restype = ctypes.c_int

    h = ctypes.c_void_p()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(),
                                   TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(h)):
        return False
    try:
        luid = LUID()
        if not advapi.LookupPrivilegeValueW(None, "SeShutdownPrivilege", ctypes.byref(luid)):
            return False
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0].Luid = luid
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
        ctypes.set_last_error(0)
        ok = advapi.AdjustTokenPrivileges(h, 0, ctypes.byref(tp), 0, None, None)
        # 这个 API 有个坑：没分到特权时它**也返回 True**，必须再看 GetLastError
        # （1300 = ERROR_NOT_ALL_ASSIGNED = 没拿到）
        return bool(ok) and ctypes.get_last_error() == 0
    finally:
        kernel.CloseHandle(h)


def _power_off_like_start_menu(force: bool = False) -> bool:
    """ExitWindowsEx(EWX_POWEROFF)，force=True 时并上 EWX_FORCE。

    **默认还是不加 EWX_FORCE** —— 有程序挡着（比如没保存的文档）时让 Windows
    停下来问，不硬杀。**但 2026-09-21 起加了一条**：光靠这个「等用户点」的行为，
    远程关机是永远关不掉的（Cherry Studio 注册了关机阻断 → Windows 停在
    「这些应用阻止关机」那一页等人点，而人根本不在机器前）。
    所以调用方 `/api/shutdown` 先探测阻断者、让用户在手机上确认，确认过了才传
    force=True 走这里。**别把默认值改成 True** —— 那等于绕过了那道确认。

    注意这个调用是异步的：返回 True 只代表「关机已启动」。
    """
    if not _enable_shutdown_privilege():
        log("[!] 没能打开 SeShutdownPrivilege，退回 shutdown.exe")
        return False
    flags = EWX_POWEROFF | (EWX_FORCE if force else 0)
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.ExitWindowsEx.argtypes = [ctypes.c_uint, ctypes.c_ulong]
        user32.ExitWindowsEx.restype = ctypes.c_int
        ctypes.set_last_error(0)
        ok = user32.ExitWindowsEx(flags, SHTDN_REASON_FLAG_PLANNED)
        if not ok:
            log(f"[!] ExitWindowsEx(flags=0x{flags:x}) 失败，错误码 {ctypes.get_last_error()}")
            return False
        return True
    except Exception as e:
        log(f"[!] ExitWindowsEx 异常：{e}")
        return False


def _suspend_now() -> bool:
    """SetSuspendState(FALSE, FALSE, FALSE) —— 进入 S3 待机（**不是**休眠）。

    **三个参数都必须是 FALSE，别去抄 rundll32 那条命令。**
    `rundll32 powrprof.dll,SetSuspendState 0,1,0` 有名的「说要睡眠结果休眠了」，毛病不在
    API：rundll32 把 "0" 当**字符串指针**传，非 NULL 就是 TRUE，于是三个参数全变 TRUE → 休眠。
    自己用 ctypes 传真 BOOLEAN 就没这个问题。本机 `powercfg /a`：S3 可用、休眠可用、
    **混合睡眠不受支持**（虚拟机监控程序挡的）→ FALSE 就是干净的 S3。

    返回 True 只代表「睡眠请求已被受理」；**这个调用会阻塞到系统真正睡下去**，
    所以调用方要先 log 再调（唤醒后那行 log 才是它返回的地方）。
    """
    try:
        powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
        powrprof.SetSuspendState.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
        powrprof.SetSuspendState.restype = ctypes.c_ubyte
        ctypes.set_last_error(0)
        ok = powrprof.SetSuspendState(0, 0, 0)      # (bHibernate, bForce, bWakeupEventsDisabled)
        if not ok:
            log(f"[!] SetSuspendState 失败，错误码 {ctypes.get_last_error()}")
            return False
        return True
    except Exception as e:
        log(f"[!] SetSuspendState 异常：{e}")
        return False


def sleep_pc(reason: str = "手机网页") -> None:
    """让这台电脑进入睡眠(S3)。**管理器不会死**，唤醒后原样接着跑。

    这是「远程关机」的正解：真关机等于要人跑到机器前输 PIN（见上面那段注释），睡眠不用。
    """
    log(f"收到睡眠指令（{reason}），正在准备睡眠")
    try:
        time.sleep(1.0)             # 先等 HTTP 响应回到手机，否则页面只会看到「请求失败」

        # **睡前必须先卸模型**：S3 会把 GPU 一起断电，唤醒后 CUDA 上下文一定失效。
        # 留着一个上下文已死的 llama-server 比卸干净更糟 —— 它多半是卡住而不是干脆报错，
        # 管理器还以为模型在、不会再重载，用户看到的就是「唤醒了但聊不了」。
        # 卸掉之后，唤醒后第一条请求会按需重载（10~15 秒），路径是验证过的。
        RT.unload("睡前卸载（S3 会让 CUDA 上下文失效）")

        # SetSuspendState 要 SE_SHUTDOWN_NAME；本机普通权限就开得上（不需要管理员）
        if not _enable_shutdown_privilege():
            log("[!] 没能打开 SeShutdownPrivilege，放弃睡眠")
            return
        log("正在进入睡眠（S3）—— 管理器会一起冻住，唤醒后原样继续，不用输 PIN")
        if _suspend_now():
            # 到这里说明机器已经醒过来了（SetSuspendState 睡过去才返回）
            log("已从睡眠中恢复，管理器继续运行")
        else:
            log("[x] 睡眠失败，管理器还在跑（可能是唤醒事件被禁用或有程序拦着）")
    except Exception as e:
        log(f"[x] 睡眠异常：{e}")


def shutdown_pc(reason: str = "手机网页", force: bool = False) -> None:
    """完全关掉这台电脑。管理器会跟着一起死，所以这是本次运行的最后一件事。

    **这是备选路径，不是主路** —— 真关机之后必须有人到机器前输 PIN 才能进系统，
    而管理器自启挂在登录上，所以关机后 80/8091 不会自己起来，远程也就断了。
    要远程用请走 sleep_pc()（手机网页上的红色「睡眠」按钮）。这里保留它只是为了
    「确实要断电」的场合（比如要搬机器）。改这里之前先读上面那段注释。

    force 由 `/api/shutdown` 传进来：用户已经在手机上看见「谁在挡着」并点了确认。
    没传 force 却走到这里，只可能是**探测时没有阻断者**（探测到就返回了，不会起这个线程）。

    **故意不先卸载模型**：关机时 Windows 会把引擎和管理器一起收走，先卸载只是白等
    5~10 秒。（睡眠就不能这么干，GPU 会掉电，见 sleep_pc。）
    """
    log(f"收到关机指令（{reason}）{'，强制' if force else ''}，正在关机")
    try:
        time.sleep(1.0)             # 先等 HTTP 响应回到手机，否则页面只会看到「请求失败」

        if _power_off_like_start_menu(force):
            log("关机命令已下达（ExitWindowsEx / EWX_POWEROFF%s）—— 下次开机会停在登录界面"
                % (" + EWX_FORCE" if force else ""))
            return

        log("改用 shutdown.exe 兜底")
        exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "shutdown.exe"
        r = subprocess.run(
            [str(exe) if exe.exists() else "shutdown",
             "/s", "/t", "3"] + (["/f"] if force else [])
             + ["/c", "模型管理器：远程关机"],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW)
        if r.returncode == 0:
            log("关机命令已下达（shutdown /s）")
        else:
            log(f"[x] 关机失败（{r.returncode}）：{(r.stderr or r.stdout or '').strip()}")
    except Exception as e:
        log(f"[x] 关机命令异常：{e}")


def parse_args():
    ap = argparse.ArgumentParser(
        prog="model_manager",
        description="本地模型管理器 —— 手机上切换电脑上的大模型",
        epilog="端口默认 80/8091/8092。换端口主要为了调试，免得打扰正在跑的正式实例。")
    ap.add_argument("--console-port", type=int, default=MANAGER_PORT)
    ap.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    ap.add_argument("--engine-port", type=int, default=ENGINE_PORT)
    ap.add_argument("--takeover", action="store_true",
                    help="启动时杀掉所有 llama-server 并接管 8091")
    ap.add_argument("--restart-wait", type=int, default=0, metavar="PORT",
                    help="重启内部用：先等这个端口空出来再启动（网页的「重启」按钮会带）")
    return ap.parse_args()


def main():
    a = parse_args()
    global MANAGER_PORT, PROXY_PORT, ENGINE_PORT
    MANAGER_PORT, PROXY_PORT, ENGINE_PORT = a.console_port, a.proxy_port, a.engine_port

    LOGDIR.mkdir(parents=True, exist_ok=True)

    if a.restart_wait:
        # 「重启」按钮起的新实例走这条路：先等旧进程放开端口。不等的话下面那句
        # port_busy 会把自己吓退（看到旧实例还占着 80 就以为已经有人在跑了）。
        log(f"重启：等端口 {a.restart_wait} 空出来…")
        if not wait_port_free(a.restart_wait):
            log(f"[x] 等了 30 秒 {a.restart_wait} 还被占着 —— 放弃启动")
            return 1
        log("重启：端口已空，继续启动")

    RT.models = discover_models()
    write_registry(RT.models)
    log(f"发现 {len(RT.models)} 个模型")

    GPU.start()                    # 后台采 GPU 功耗，给页面画曲线

    if port_busy(MANAGER_PORT):
        log(f"[x] 统一入口端口 {MANAGER_PORT} 已被占用 —— 管理器可能已经在跑了")
        return 1

    if a.takeover:
        n = kill_engines()
        RT._ext_cache = (0.0, [])
        time.sleep(1)
        log(f"--takeover：已清理 {n} 个 llama-server 进程")
    else:
        # 认领已经在跑的引擎 —— 重启管理器不会把模型踹掉
        RT.adopt_existing()

    # 8091 被手动启动的模型占着时，不去打断它 —— 先起控制台，接管交给用户决定
    if port_busy(PROXY_PORT):
        ext = RT.external_pids()
        if ext:
            log(f"[!] {PROXY_PORT} 被一个正在运行的 llama-server 占着（PID {', '.join(map(str, ext))}）")
            log("    这多半是你手动跑着模型在用。管理器不会去动它。")
            log("    现在以「只读控制台」模式启动：80 控制界面可用，8091 兼容代理暂不可用。")
            log("    想收编：手机上点「接管」，或者加 --takeover 参数重启管理器。")
        else:
            log(f"[!] {PROXY_PORT} 被别的程序占着，兼容代理起不来。主入口 80 照常可用。")
    else:
        start_proxy()

    # 同一个主机名根路径提供控制台，/v1 路由到 OpenAI 兼容代理。
    # 旧 8090 控制台端口不再监听；8091 保留给既有客户端。
    console = Server(("0.0.0.0", MANAGER_PORT), ProxyHandler)
    threading.Thread(target=console.serve_forever, daemon=True).start()

    ip = lan_ip()

    print("=" * 64)
    print("  本地模型管理器已启动")
    print("=" * 64)
    print(f"  手机控制台   http://{LAN_HOSTNAME}/?k={API_KEY}")
    print(f"  当前 IP 备用 http://{ip}/?k={API_KEY}")
    if PROXY_OK:
        print(f"  模型软件 API http://{LAN_HOSTNAME}/v1    (支持按需自动加载)")
        print(f"  兼容入口     http://{LAN_HOSTNAME}:{PROXY_PORT}/v1")
    else:
        print(f"  兼容入口     未启动 —— {PROXY_PORT} 被占。")
        print(f"               主入口 /v1 仍可通过 80 端口访问。")
    print(f"  内部引擎端口 {ENGINE_PORT}  (llama-server)")
    print()
    print(f"  发现 {len(RT.models)} 个模型。手机浏览器打开上面那个地址就能切换。")
    print("  按 Ctrl+C 退出")
    print("=" * 64)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("收到 Ctrl+C，正在停止…")
        try:
            RT.unload("管理器退出")
        except Exception:
            pass
        console.shutdown()
        if _proxy_server:
            _proxy_server.shutdown()
        log("已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
