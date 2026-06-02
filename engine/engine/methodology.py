"""Methodology-reinforcement / anti-drift text catalog + session state.

Background: long-horizon utov sessions (the reference target was the canonical
example) drift from "ledger-driven" to "context-driven" — the agent
starts trusting the last big number it saw, bypassing the verifier,
calling things "确认" without a subject, and treating un-ledger-promoted
experiments as success evidence. Human reconciliation kept pulling the
session back; this module turns reconciliation from after-the-fact
clean-up into runtime self-check pressure.

Two surfaces consumers see:

  - **footer**  — short checklist appended to EVERY tool result. Cheap;
    the agent skims it on every reply.
  - **card**    — full methodology card with the 5 rules + current
    session state. Injected every ``periodic_interval`` steps (default
    15) so the agent gets a real "come back to the methodology" moment.
  - **prompts** — context-sensitive reverse questions (evidence_class,
    contradiction, high-rate success, ...).
  - **alerts**  — runtime interceptions (verifier bypass count, ledger
    M2 violation, forbidden keyword).

Everything else (text strings, knob defaults) is data — kept here so
revisions don't require touching the wrapper plumbing.

Linkage:
  - PLAN §12.3 anti-drift injection (this is its runtime form)
  - mechanism_improvements.md M1 / M2 / M3 / M5 / M6 / M8
  - capability_request.md §P1-1 (high-number success prompt) / §P1-2
    (contradiction-driven prompt)
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Text catalog — single source of truth for every line shown to the agent.
# Translators / editors touch this file only.
# ---------------------------------------------------------------------------

FOOTER_TEMPLATE = (
    "[clark-utov ✓ {operation} 完成]\n"
    "方法论自检:\n"
    "  □ 入口出口锁死了吗?\n"
    "  □ 状态写进账本了吗?\n"
    "  □ 错误发现了要回退吗?\n"
    "  □ 失败时用差分定位了吗?\n"
    "  □ 从干净 checkpoint 出发了吗?\n"
    "\n"
    "下一步前请选:\n"
    "  [继续] - 沿当前路径推进\n"
    "  [回退] - 上一个 finding 有问题,需回滚\n"
    "  [盘整] - 触发强制状态盘整\n"
    "  [登记机制改进] - 发现 utov 能力缺口"
)

PERIODIC_CARD_TEMPLATE = (
    "[clark-utov 周期性纪律提醒 · 第 {step} 步]\n"
    "\n"
    "方法论五条:\n"
    "1. 锁死入口出口\n"
    "2. 记好账本\n"
    "3. 错了回退\n"
    "4. 差分定位错点\n"
    "5. 重新出发\n"
    "\n"
    "当前 session 状态:\n"
    "- 活跃 finding 数: {active_findings}\n"
    "- 未闭合 hyp 数: {open_hyps}\n"
    "- 回溯次数: {backtracks}\n"
    "- 最近 N 步主要操作类型: {recent_ops}\n"
    "\n"
    "自查:你最近 N 步有违反任何一条吗?如有,先盘整再继续。"
)

# Context-sensitive prompt text — single line each. Keys match the
# PromptKind enum below.
PROMPT_TEXTS: dict[str, str] = {
    "produced_verdict": (
        "需要我帮你登记进账本吗?这个结论的 evidence_class"
        "(A 硬证据 / B 排除法 / C 推测)是什么?"
    ),
    "contradicts_finding": (
        "检测到和 finding#{finding_id} 矛盾,需要我帮你触发盘整吗?"
    ),
    "high_number_success": (
        "这是 evidence_class A 还是 pending_review?"
        "需要我帮你做中间态确认吗?"
    ),
    "repeated_failures": (
        "建议盘整,需要我帮你查最近的 Γ checkpoint 吗?"
    ),
    "non_utov_path": (
        "检测到非 utov 路径,需要我帮你查 utov 是否有对应能力吗?"
    ),
    "multi_candidate": (
        "提醒:M-R2 禁止多候选凑数,选定的源是哪一个?证据是什么?"
    ),
}

# Runtime alert text. Always raised in the result so the agent must
# acknowledge before continuing.
ALERT_TEXTS: dict[str, str] = {
    "verifier_bypass": (
        "已检测到第 {count} 次绕过 verifier 的判定。"
        "强制盘整建议触发 — 在响应中明确确认例外或调用 checkpoint() 才能继续。"
    ),
    "unledgered_reference": (
        "拒绝调用 {method}:payload 含未入账本数据(experiment / unpromoted)。"
        "先调用 promote_to_finding 或在 reason 里加 `--allow-unpromoted` 例外。"
    ),
    "no_recent_checkpoint": (
        "距上次盘整 {steps} 步且期间 {failures} 次失败;主动建议盘整。"
    ),
    "forbidden_keyword": (
        "检测到关键词模式 {keyword!r} — 走 utov 对应能力"
        "(如 localize_divergence / hook_sanity)而非手工流程。"
    ),
}


# ---------------------------------------------------------------------------
# Configuration & state
# ---------------------------------------------------------------------------


DEFAULT_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "sign 回填",
    "手动 dump",
    "凭印象",
    "凭直觉",
    "估摸",
)

# Operation methods that count as "bypassing verifier" for the
# verifier-bypass alert (P1 violation #1).
DEFAULT_BYPASS_METHODS: tuple[str, ...] = (
    "override_verdict",
    "force_status",
)

# Operation methods that produce a verdict / verification outcome. Used
# for the "你刚产出一个判定" prompt.
DEFAULT_VERDICT_METHODS: tuple[str, ...] = (
    "promote_to_finding",
    "inject_finding",
    "override_verdict",
)

# Methods whose result objects carry numeric pass-rates worth scrutiny.
DEFAULT_NUMERIC_RESULT_METHODS: tuple[str, ...] = (
    "verify_plugin_findings",
    "verify_handler_binops",
    "verify_handler_unaries",
    "verify_handler_imm_binops",
    "verify_handler_extended_binops",
    "verify_handler_bfx",
    "verify_handler_ch_idioms",
    "verify_handler_maj_idioms",
    "verify_triton_simplifications",
    "verify_sigma_idioms",
    "verify_algorithm_templates",
    "preprocess_batch",
)


@dataclass(slots=True)
class MethodologyConfig:
    """Tunable knobs. Defaults align with the spec; CLI / serve_mcp can
    override per session."""
    enabled: bool = True
    periodic_interval: int = 15
    recent_ops_window: int = 15
    failure_streak_threshold: int = 3
    steps_since_checkpoint_warn: int = 25
    bypass_alert_threshold: int = 3
    high_success_floor: float = 0.99
    forbidden_keywords: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_FORBIDDEN_KEYWORDS,
    )
    bypass_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_BYPASS_METHODS,
    )
    verdict_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_VERDICT_METHODS,
    )
    numeric_result_methods: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_NUMERIC_RESULT_METHODS,
    )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MethodologyConfig":
        """Build a config from environment overrides. ``UTOV_METHODOLOGY``
        set to ``off`` / ``0`` / ``false`` disables the whole layer."""
        src = env if env is not None else os.environ
        cfg = cls()
        flag = (src.get("UTOV_METHODOLOGY") or "").strip().lower()
        if flag in ("off", "0", "false", "no"):
            cfg.enabled = False
        for env_key, attr, cast in (
            ("UTOV_METHODOLOGY_INTERVAL",        "periodic_interval",          int),
            ("UTOV_METHODOLOGY_RECENT_WINDOW",   "recent_ops_window",          int),
            ("UTOV_METHODOLOGY_FAILURE_STREAK",  "failure_streak_threshold",   int),
            ("UTOV_METHODOLOGY_CHECKPOINT_WARN", "steps_since_checkpoint_warn", int),
            ("UTOV_METHODOLOGY_BYPASS_TH",       "bypass_alert_threshold",     int),
        ):
            v = src.get(env_key)
            if v is None:
                continue
            try:
                setattr(cfg, attr, cast(v))
            except ValueError:
                continue
        return cfg


@dataclass(slots=True)
class MethodologyState:
    """Per-session mutable counters."""
    step_count: int = 0
    bypass_count: int = 0
    backtrack_count: int = 0
    failures_since_checkpoint: int = 0
    steps_since_checkpoint: int = 0
    last_periodic_card_at: int = 0
    recent_ops: deque[str] = field(default_factory=deque)
    recent_failures: deque[str] = field(default_factory=deque)
    # When a violation was raised at step S, agent must acknowledge
    # before the next step proceeds. We store the acknowledgement key.
    pending_ack: str | None = None

    def push_op(self, method: str, window: int) -> None:
        self.recent_ops.append(method)
        while len(self.recent_ops) > window:
            self.recent_ops.popleft()

    def push_failure(self, method: str, window: int) -> None:
        self.recent_failures.append(method)
        while len(self.recent_failures) > window:
            self.recent_failures.popleft()
        self.failures_since_checkpoint += 1

    def reset_failures(self) -> None:
        self.recent_failures.clear()
        self.failures_since_checkpoint = 0
        self.steps_since_checkpoint = 0


# ---------------------------------------------------------------------------
# Renderers — pure functions that turn state into the actual strings.
# ---------------------------------------------------------------------------


def render_footer(method: str) -> str:
    """The fixed checklist appended to every tool result."""
    return FOOTER_TEMPLATE.format(operation=method)


def render_periodic_card(
    state: MethodologyState,
    *,
    step: int,
    active_findings: int,
    open_hyps: int,
) -> str:
    """Full methodology card with session telemetry."""
    if state.recent_ops:
        from collections import Counter
        top = Counter(state.recent_ops).most_common(3)
        recent_summary = ", ".join(f"{op}×{n}" for op, n in top)
    else:
        recent_summary = "(none)"
    return PERIODIC_CARD_TEMPLATE.format(
        step=step,
        active_findings=active_findings,
        open_hyps=open_hyps,
        backtracks=state.backtrack_count,
        recent_ops=recent_summary,
    )


def render_prompt(kind: str, **fmt: Any) -> str:
    """Format a context-sensitive prompt by its kind."""
    tmpl = PROMPT_TEXTS.get(kind)
    if tmpl is None:
        return ""
    return tmpl.format(**fmt)


def render_alert(kind: str, **fmt: Any) -> str:
    """Format an interception alert by its kind."""
    tmpl = ALERT_TEXTS.get(kind)
    if tmpl is None:
        return ""
    return tmpl.format(**fmt)
