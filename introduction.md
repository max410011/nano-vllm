# Nano-vLLM Introduction

## 1. 專案概述

Nano-vLLM 是一個輕量級的 vLLM 實現，用約 1,200 行 Python 代碼從零構建。它提供了與 vLLM 相當的推理速度，同時保持代碼簡潔易讀。

### 1.1 核心特性
- 🚀 **高效離線推理** - 與 vLLM 相當的推理速度
- 📖 **可讀性高的代碼庫** - ~1,200 行乾淨的 Python 實現
- ⚡ **優化套件** - Prefix Caching、Tensor Parallelism、Torch 編譯、CUDA Graph 等

---

## 2. 專案架構

```
nanovllm/
├── __init__.py          # 導出 LLM, SamplingParams
├── config.py            # 全局配置類
├── llm.py               # LLM 入口類（繼承 LLMEngine）
├── sampling_params.py   # 採樣參數配置
├── engine/              # 推理引擎核心
│   ├── llm_engine.py    # 主引擎：請求管理、調度、生成循環
│   ├── model_runner.py  # 模型執行：前向傳播、KV Cache、CUDA Graph
│   ├── scheduler.py     # 批次調度：Prefill/Decode 調度
│   ├── sequence.py      # 序列狀態管理
│   └── block_manager.py # KV Cache Block 管理與 Prefix Caching
├── layers/              # 模型層組件
│   ├── attention.py     # 注意力機制 + KV Cache 存取
│   ├── linear.py        # 線性層（含 Tensor Parallelism）
│   ├── rotary_embedding.py  # RoPE 位置編碼
│   ├── layernorm.py     # RMSNorm 實現
│   ├── activation.py    # SiLU 激活函數
│   ├── embed_head.py    # Embedding 與 LM Head
│   └── sampler.py       # Token 採樣器
├── models/              # 模型實現
│   └── qwen3.py         # Qwen3 模型架構
└── utils/               # 工具函數
    ├── context.py       # 全局上下文（prefill/decode 狀態）
    └── loader.py        # 模型權重加載
```

---

## 3. 核心組件詳解

### 3.1 Config (`config.py`)

全局配置類，管理模型路徑、批次大小、GPU 記憶體利用率等參數。

```python
@dataclass
class Config:
    model: str                        # 模型路徑
    max_num_batched_tokens: int       # 最大批次 token 數（預設 16384）
    max_num_seqs: int                 # 最大並發序列數（預設 512）
    max_model_len: int                # 最大上下文長度（預設 4096）
    gpu_memory_utilization: float     # GPU 記憶體利用率（預設 0.9）
    tensor_parallel_size: int         # Tensor 並行數（預設 1）
    enforce_eager: bool               # 是否禁用 CUDA Graph（預設 False）
    kvcache_block_size: int           # KV Cache Block 大小（預設 256）
```

### 3.2 LLMEngine (`engine/llm_engine.py`)

主推理引擎，負責：
- **請求管理**：接收 prompt，創建 Sequence 對象
- **多進程管理**：Tensor Parallelism 時創建子進程
- **生成循環**：調度 → 執行 → 後處理的主循環
- **Throughput 監控**：實時顯示 Prefill/Decode 速度

關鍵方法：
- `add_request()`: 添加新請求到調度器
- `step()`: 執行一步推理（prefill 或 decode）
- `generate()`: 批量生成完整輸出

### 3.3 ModelRunner (`engine/model_runner.py`)

模型執行器，負責：
- **模型加載**：初始化模型、分配 KV Cache
- **輸入準備**：構建 prefill/decode 的輸入張量
- **CUDA Graph**：捕獲和重放計算圖以加速 decode
- **分佈式通信**：使用 SharedMemory 在多 GPU 間同步

關鍵功能：
- `prepare_prefill()`: 準備 prefill 階段輸入（變長序列）
- `prepare_decode()`: 準備 decode 階段輸入（單 token）
- `capture_cudagraph()`: 捕獲不同 batch size 的 CUDA Graph

### 3.4 Scheduler (`engine/scheduler.py`)

批次調度器，實現 **Prefill-first** 策略：
- 優先調度等待中的 prefill 請求
- Prefill 完成後進入 decode 階段
- 支援 **搶占 (Preemption)**：記憶體不足時暫停序列

### 3.5 BlockManager (`engine/block_manager.py`)

KV Cache 記憶體管理器，實現：
- **Paged KV Cache**：將 KV Cache 分成固定大小的 Block
- **Prefix Caching**：使用 hash 重用相同前綴的 KV Cache
- **動態分配/回收**：按需分配和釋放 Block

### 3.6 Attention (`layers/attention.py`)

注意力機制實現：
- 使用 **FlashAttention** 進行高效計算
- `flash_attn_varlen_func`: Prefill 階段（變長序列）
- `flash_attn_with_kvcache`: Decode 階段（使用 KV Cache）
- **Triton Kernel**: 高效寫入 KV Cache

### 3.7 Linear Layers (`layers/linear.py`)

支援 Tensor Parallelism 的線性層：
- `ColumnParallelLinear`: 輸出維度分割
- `RowParallelLinear`: 輸入維度分割（含 all-reduce）
- `QKVParallelLinear`: Q/K/V 合併投影
- `MergedColumnParallelLinear`: Gate/Up 合併投影

### 3.8 Context (`utils/context.py`)

全局上下文管理：
- 存儲當前批次的 prefill/decode 狀態
- 提供 `cu_seqlens`、`slot_mapping`、`block_tables` 等張量

---

## 4. 推理流程

### 4.1 初始化流程
1. 創建 `Config`，加載 HuggingFace 配置
2. 啟動子進程（Tensor Parallelism）
3. 初始化 `ModelRunner`：加載模型、預熱、分配 KV Cache
4. 捕獲 CUDA Graph（如未啟用 eager mode）

### 4.2 生成流程
```
add_request() → Scheduler.add()
     ↓
while not finished:
     │
     ├─ Scheduler.schedule() → 選擇 prefill 或 decode 批次
     │
     ├─ ModelRunner.run() → 執行模型前向傳播
     │   ├─ prepare_prefill/decode() → 構建輸入
     │   ├─ run_model() → 執行模型（或重放 CUDA Graph）
     │   └─ sampler() → 採樣下一個 token
     │
     └─ Scheduler.postprocess() → 更新序列狀態
```

---

## 5. 關鍵優化技術

| 優化技術 | 實現位置 | 說明 |
|---------|---------|------|
| **Paged KV Cache** | `block_manager.py` | 分塊管理 KV Cache，避免記憶體碎片 |
| **Prefix Caching** | `block_manager.py` | Hash-based 重用相同前綴的 KV Cache |
| **FlashAttention** | `attention.py` | 高效融合注意力計算 |
| **CUDA Graph** | `model_runner.py` | 捕獲和重放 decode 計算圖 |
| **Tensor Parallelism** | `linear.py`, `model_runner.py` | 多 GPU 並行推理 |
| **Torch Compile** | `rotary_embedding.py`, `sampler.py` | JIT 編譯加速 |
| **Triton Kernel** | `attention.py` | 高效 KV Cache 寫入 |

---

## 6. 與 xKV 整合的切入點

整合 xKV 壓縮需要修改以下組件：

1. **Config**: 添加 xKV 相關配置參數
2. **Attention**: 在 prefill 時存儲 pre-RoPE KV，支援壓縮後寫回
3. **ModelRunner**: 管理 xKV Cache Manager，觸發壓縮
4. **Scheduler/BlockManager**: 可能需要調整以支援壓縮後的 KV 大小

---

