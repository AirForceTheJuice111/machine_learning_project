from __future__ import annotations


TARGET_DESIGN_GUIDANCE: dict[str, str] = {
    "actual_boost_clock_mhz": (
        "使用算术密集型长时间 kernel，结合 clock64 与 host 计时估算稳定频率，"
        "不要直接读取规格表。"
    ),
    "bank_conflict_penalty_cycles": (
        "使用 shared memory 单 warp 访问，对比无冲突 stride=1 与严重冲突 stride=32 的延迟差。"
    ),
    "dram_latency_cycles": (
        "使用 pointer chasing 与大 working set，避免并行隐藏延迟，从每步依赖链中估算 DRAM 延迟。"
    ),
    "l2_latency_cycles": (
        "使用 pointer chasing，但把 working set 控制在 L2 容量附近，预热后测量稳定的 L2 命中延迟。"
    ),
    "l1_latency_cycles": (
        "使用极小 working set 或重复访问固定位置，尽可能让访问留在 L1，再估算单次访问延迟。"
    ),
    "l2_cache_capacity_kb": (
        "通过 working set sweep 逐步扩大访问集合，观察 latency cliff 以推断 L2 容量。"
    ),
    "global_memory_bandwidth_gbps": (
        "使用大数组 streaming load/store 或 copy benchmark，通过总传输字节数除以稳定时间估算带宽。"
    ),
    "shared_memory_bandwidth_gbps": (
        "使用 tight shared-memory load/store benchmark，在足够迭代下估算共享内存吞吐。"
    ),
    "max_shmem_per_block_kb": (
        "使用动态 shared memory kernel，逐步增加每 block 申请量直到 launch 失败，从而得到上限。"
    ),
}


DEFAULT_TARGET_DESIGN_GUIDANCE = (
    "先根据指标名判断其属于 latency、bandwidth、capacity、frequency 或 resource limit，"
    "再设计最小化 micro-benchmark。"
)


def build_target_design_guidance(targets: list[str]) -> str:
    lines = ["不同探测目标的 CUDA C++ 设计思路如下："]
    for target in targets:
        idea = TARGET_DESIGN_GUIDANCE.get(target, DEFAULT_TARGET_DESIGN_GUIDANCE)
        lines.append(f"- {target}: {idea}")
    lines.append("你可以参考这些思路自主写代码，但不要照抄固定模板，仍需根据当前 GPU 与目标进行调整。")
    return "\n".join(lines)
