from __future__ import annotations


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
