from __future__ import annotations


TARGET_FAMILY_MAP: dict[str, str] = {
    "l1_latency_cycles": "latency_family",
    "l2_latency_cycles": "latency_family",
    "dram_latency_cycles": "latency_family",
    "l2_cache_capacity_kb": "latency_family",
    "l2_cache_capacity_mb": "latency_family",
    "global_memory_bandwidth_gbps": "bandwidth_family",
    "global_mem_peak_gbps": "bandwidth_family",
    "vram_peak_gbps": "bandwidth_family",
    "shared_memory_bandwidth_gbps": "bandwidth_family",
    "shared_mem_peak_gbps": "bandwidth_family",
    "actual_boost_clock_mhz": "clock_family",
    "bank_conflict_penalty_cycles": "bank_conflict_family",
    "max_shmem_per_block_kb": "resource_limit_family",
}


FAMILY_DESIGN_CONSTRAINTS: dict[str, dict[str, str]] = {
    "latency_family": {
        "goal": "用一份共享的 latency/working-set benchmark 同时覆盖 L1、L2、DRAM latency 与 L2 容量 cliff。",
        "must": "优先生成一个支持多个 mode 的 probe，例如 l1、l2、dram、capacity_sweep；应尽量复用同一份 .cu 和同一二进制。",
        "avoid": "不要为同一家族里的每个 target 各写一份完全独立的 benchmark，也不要用吞吐型实验代替 latency 实验。",
    },
    "bandwidth_family": {
        "goal": "用一份共享的 bandwidth benchmark 覆盖 global/VRAM/shared memory 的峰值带宽测量。",
        "must": "优先生成一个支持多个 mode 的 probe，例如 global、vram、shared；同一二进制通过不同参数切换测量路径。",
        "avoid": "不要把缓存命中主导的小数据测试结果当作峰值带宽，也不要把 shared memory 和 global memory 路径混在同一个 mode 里。",
    },
    "clock_family": {
        "goal": "用一份 compute-heavy benchmark 估算持续负载下的实际 boost 频率。",
        "must": "单独生成适合持续算术负载的 probe，必要时支持 warmup 与 measured 两种 mode。",
        "avoid": "不要直接查规格表，不要把不合理量级的 cycles/time 结果当成真实核心频率。",
    },
    "bank_conflict_family": {
        "goal": "用一份 shared-memory benchmark 量化 bank conflict penalty。",
        "must": "优先在同一二进制中支持 conflict_free 与 conflict_heavy 两种 mode，再比较差值。",
        "avoid": "不要把完全不同的 kernel 混在一起导致无法归因。",
    },
    "resource_limit_family": {
        "goal": "用一份资源探测 benchmark 测量 block 级共享内存上限。",
        "must": "生成支持参数扫描的 probe，在同一程序中逐步提高动态 shared memory 请求量。",
        "avoid": "不要只读取 API 报告值，也不要把 unrelated benchmark 混进来。",
    },
}


TARGET_DESIGN_CONSTRAINTS: dict[str, dict[str, str]] = {
    "actual_boost_clock_mhz": {
        "goal": "在持续算术负载下估算稳定核心频率，而不是读取规格表或静态属性。",
        "must": "设计长时间 arithmetic-heavy kernel，并结合 device 侧 cycle 计数与 host 侧时间换算 MHz。",
        "avoid": "不要直接依赖 cudaGetDeviceProperties、规格表或固定型号经验值。",
    },
    "bank_conflict_penalty_cycles": {
        "goal": "量化 shared memory bank conflict 相对无冲突访问的额外代价。",
        "must": "至少比较一个 conflict-free 访问模式与一个高冲突访问模式，并从差值估算 penalty。",
        "avoid": "不要把总 kernel 时间直接当作单次 bank conflict 代价；要尽量隔离访存模式差异。",
    },
    "dram_latency_cycles": {
        "goal": "测量真实 DRAM 访问延迟。",
        "must": "使用依赖链 pointer chasing 和足够大的 working set，避免并行隐藏延迟与预取干扰。",
        "avoid": "不要用吞吐型 streaming benchmark 代替 latency probe。",
    },
    "l2_latency_cycles": {
        "goal": "测量主要命中 L2 时的访问延迟。",
        "must": "使用 pointer chasing，并把 working set 控制在接近 L2 可容纳范围，必要时先预热。",
        "avoid": "不要让 working set 小到明显落在 L1，也不要大到稳定落入 DRAM。",
    },
    "l1_latency_cycles": {
        "goal": "测量主要命中 L1 时的访问延迟。",
        "must": "让访问集合足够小并重复命中同一小片区域，尽量提高 L1 hit 概率。",
        "avoid": "不要把首次冷启动访问或大工作集访问的结果当作 L1 latency。",
    },
    "l2_cache_capacity_kb": {
        "goal": "识别 latency-size 曲线中的 cache cliff 来推断 L2 容量。",
        "must": "做 working set sweep，并输出或总结延迟随大小变化的拐点证据。",
        "avoid": "不要直接查设备参数或假设标准显卡规格。",
    },
    "l2_cache_capacity_mb": {
        "goal": "识别 latency-size 曲线中的 cache cliff 来推断 L2 容量，并以 MB 输出。",
        "must": "做 working set sweep，先估算 L2 容量，再把容量单位换算为 MB 后输出。",
        "avoid": "不要直接查设备参数或假设标准显卡规格，也不要输出错误单位。",
    },
    "global_memory_bandwidth_gbps": {
        "goal": "估算当前环境下可达到的 global memory 有效带宽。",
        "must": "使用大数组 streaming load/store/copy 类 benchmark，并按总传输字节数除以稳定耗时换算带宽。",
        "avoid": "不要把缓存命中主导的小数据测试结果当作 DRAM 带宽。",
    },
    "global_mem_peak_gbps": {
        "goal": "估算当前环境下可达到的 global memory 峰值带宽。",
        "must": "使用大数组 streaming load/store/copy 类 benchmark，并按总传输字节数除以稳定耗时换算 Gbps。",
        "avoid": "不要把缓存命中主导的小数据测试结果当作峰值带宽。",
    },
    "vram_peak_gbps": {
        "goal": "估算 VRAM/DRAM 路径的峰值带宽。",
        "must": "使用能显著落到外部显存的数据规模与访问模式，按稳定传输字节数和耗时换算 Gbps。",
        "avoid": "不要让结果主要反映缓存或 shared memory 吞吐。",
    },
    "shared_memory_bandwidth_gbps": {
        "goal": "估算 shared memory 的有效吞吐上限。",
        "must": "构造高强度 shared-memory load/store 循环，并保证迭代足够多以压低 launch 开销影响。",
        "avoid": "不要混入大量 global memory 流量，否则结果不再代表 shared memory 带宽。",
    },
    "shared_mem_peak_gbps": {
        "goal": "估算 shared memory 的峰值吞吐。",
        "must": "构造高强度 shared-memory load/store 循环，并用稳定区间换算 Gbps。",
        "avoid": "不要混入大量 global memory 流量，也不要把 launch 开销主导的结果当成峰值。",
    },
    "max_shmem_per_block_kb": {
        "goal": "探测每个 block 可申请的动态 shared memory 上限。",
        "must": "逐步提高每 block 动态 shared memory 申请量，观察 launch 成功与失败边界。",
        "avoid": "不要只读取 API 报告值；在被限制或虚拟化环境下它可能不可靠。",
    },
}


DEFAULT_TARGET_DESIGN_CONSTRAINTS = {
    "goal": "根据 target 名判断其属于 latency、bandwidth、capacity、frequency 或 resource limit，并为该目标单独设计 probe。",
    "must": "先明确可观测量、测量公式与干扰项，再写最小化 micro-benchmark。",
    "avoid": "不要套用无关模板，不要直接查规格表，也不要用单个泛化实验同时回答多个 target。",
}


def build_target_design_guidance(targets: list[str]) -> str:
    lines = [
        "下面给出的不是代码模板，而是按 target 注入的设计约束；你必须自行决定代码结构、参数扫描方式和结果汇总方式。",
    ]
    for target in targets:
        constraints = TARGET_DESIGN_CONSTRAINTS.get(target, DEFAULT_TARGET_DESIGN_CONSTRAINTS)
        lines.append(f"[{target}]")
        lines.append(f"- 目标: {constraints['goal']}")
        lines.append(f"- 必须满足: {constraints['must']}")
        lines.append(f"- 禁止或避免: {constraints['avoid']}")
    lines.append("你需要把这些约束转化为当前 GPU 上可执行的 CUDA C++ micro-benchmark，而不是照抄固定实现。")
    return "\n".join(lines)


def infer_target_family(target: str) -> str:
    if target in TARGET_FAMILY_MAP:
        return TARGET_FAMILY_MAP[target]

    target_lower = target.lower()
    if "latency" in target_lower or "cache_capacity" in target_lower:
        return "latency_family"
    if "bandwidth" in target_lower or "peak_gbps" in target_lower:
        return "bandwidth_family"
    if "clock" in target_lower or "mhz" in target_lower:
        return "clock_family"
    if "conflict" in target_lower:
        return "bank_conflict_family"
    if "shmem" in target_lower or "shared" in target_lower or "limit" in target_lower:
        return "resource_limit_family"
    return "generic_family"


def group_targets_by_family(targets: list[str]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for target in targets:
        family = infer_target_family(target)
        grouped.setdefault(family, []).append(target)
    return list(grouped.items())


def build_family_design_guidance(family: str, targets: list[str]) -> str:
    constraints = FAMILY_DESIGN_CONSTRAINTS.get(
        family,
        {
            "goal": "为这一组 target 设计可复用的共享 benchmark。",
            "must": "尽量在一份 .cu 和一个可执行程序中通过 mode/参数覆盖多个 target。",
            "avoid": "不要无必要地为每个 target 单独生成完全独立的 probe。",
        },
    )

    lines = [
        f"当前 family: {family}",
        f"- Family 目标: {constraints['goal']}",
        f"- Family 必须满足: {constraints['must']}",
        f"- Family 禁止或避免: {constraints['avoid']}",
        "",
        "当前 family 内各 target 的额外约束如下：",
    ]
    for target in targets:
        target_constraints = TARGET_DESIGN_CONSTRAINTS.get(target, DEFAULT_TARGET_DESIGN_CONSTRAINTS)
        lines.append(f"[{target}]")
        lines.append(f"- 目标: {target_constraints['goal']}")
        lines.append(f"- 必须满足: {target_constraints['must']}")
        lines.append(f"- 禁止或避免: {target_constraints['avoid']}")
    lines.append("你应优先实现一个多 mode 共享 benchmark，再用不同 mode 或参数运行同一二进制，最后拆回各个 target 的结果。")
    return "\n".join(lines)
