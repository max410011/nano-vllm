# xKV Integration Plan for Nano-vLLM

## 概述

本計劃將 xKV（跨層 KV-Cache 壓縮）和 lm_eval 整合進 nano-vllm，分為 5 個步驟：

| Step | 目標 | 驗證方式 |
|------|------|----------|
| 0 | 環境設置 | 使用 uv 建立環境，驗證 CUDA 和 nano-vllm 可正常運作 |
| 1 | 整合 lm_eval | 在 HellaSwag 等任務上測試 baseline PPL |
| 2 | 整合 xKV (基礎版) | 用 lm_eval RULER 測試壓縮後 PPL |
| 3 | 整合 xKV + Paged Attention | 用 lm_eval RULER 測試 PPL |
| 4 | 整合 xKV-sparse | 用 lm_eval 測試 PPL |

---

## Step 0: 環境設置

### 0.1 目標
- 使用 `uv` 創建隔離的虛擬環境
- 安裝所有必要依賴（包括 CUDA 相關）
- 驗證 nano-vllm 基本推理功能正常
- 創建環境設置腳本方便復現

### 0.2 需要創建的檔案
```
scripts/
├── setup_env.sh         # 環境設置腳本
├── verify_env.py        # 環境驗證腳本
```

### 0.3 環境需求
- Python 3.10-3.12
- CUDA 12.x (與 PyTorch 2.4+ 兼容)
- 依賴：torch, triton, transformers, flash-attn, xxhash, lm-eval

### 0.4 驗證項目
```python
# scripts/verify_env.py
def verify_cuda():
    """驗證 CUDA 可用性"""

def verify_flash_attn():
    """驗證 flash-attn 安裝正確"""

def verify_nano_vllm_inference():
    """驗證 nano-vllm 基本推理"""
```

### 0.5 驗收標準
- [ ] uv 虛擬環境創建成功
- [ ] 所有依賴安裝成功（無版本衝突）
- [ ] CUDA 可用且版本兼容
- [ ] flash-attn 可正常 import
- [ ] nano-vllm 可進行基本生成

---

## Step 1: 整合 lm_eval

### 1.1 目標
- 讓 nano-vllm 支援 `lm-evaluation-harness` 的標準評估介面
- 實現 `compute_prompt_logprobs()` 方法計算 log-likelihood
- https://github.com/EleutherAI/lm-evaluation-harness
- Python API	Programmatic usage with simple_evaluate()

### 1.2 需要創建的檔案
```
nanovllm/
├── eval/
│   ├── __init__.py
│   └── lm_harness.py    # NanoVLLM wrapper for lm_eval
scripts/
└── eval_lm_harness.py   # 評估腳本入口
```

### 1.3 實現細節

#### 1.3.1 添加 `compute_prompt_logprobs()` 到 LLMEngine
```python
# engine/llm_engine.py
def compute_prompt_logprobs(self, token_ids: list[int]) -> torch.Tensor:
    """計算 prompt 的 log-probabilities"""
    # 創建單一序列，執行 prefill
    # 返回 shape: (seq_len, vocab_size) 的 log_softmax 輸出
```

#### 1.3.2 實現 lm_eval 適配器 (參考 backup/eval_lm_harness.py)
- 繼承 `TemplateLM`
- 實現 `_loglikelihood_tokens()`, `loglikelihood_rolling()`, `generate_until()`

### 1.4 測試
```python
# tests/test_lm_eval_integration.py
def test_loglikelihood_basic():
    """測試 log-likelihood 計算是否正確"""
    
def test_eval_hellaswag():
    """在 HellaSwag subset 上測試評估流程"""
```

### 1.5 驗收標準
- [ ] `compute_prompt_logprobs()` 輸出 shape 正確
- [ ] lm_eval 可以成功運行 HellaSwag 評估
- [ ] baseline 模型的 accuracy 符合預期

---

## Step 2: 整合 xKV (基礎版 - 無 Paged Attention)

### 2.1 目標
- 在 prefill 階段實現跨層 KV-Cache SVD 壓縮
- 先不寫回 paged cache，直接在 attention forward 中使用壓縮後的 KV

### 2.2 需要創建的檔案
```
nanovllm/
├── xkv/
│   ├── __init__.py
│   ├── config.py        # xKVConfig, LayerGroup (參考 backup/config.py)
│   ├── compressor.py    # fake_svd, slerp_merge (參考 backup/compressor.py)
│   └── cache_manager.py # xKVCacheManager (參考 backup/cache_manager.py)
```

### 2.3 實現細節

#### 2.3.1 修改 Config
```python
# config.py - 添加 xKV 相關參數
enable_xkv: bool = False
xkv_group_size: int = 2
xkv_rank_k: int = 256
xkv_rank_v: int = 768
xkv_merge_impl: str = "svd"  # "svd" or "slerp"
```

#### 2.3.2 修改 Attention
```python
# layers/attention.py
class Attention:
    def forward(self, q, k, v, layer_idx=None, xkv_manager=None):
        # 1. 如果 xKV 啟用且是 prefill:
        #    - 存儲 pre-RoPE k, v 到 xkv_manager
        #    - 如果是 group 最後一層，觸發壓縮
        #    - 使用壓縮後的 k, v 進行 attention
        # 2. 照常執行 attention
```

#### 2.3.3 修改 Model (Qwen3Attention)
- 傳遞 `layer_idx` 和 `xkv_manager` 給 attention
- 在 RoPE 之前捕獲 k, v

### 2.4 測試
```python
# tests/test_xkv_basic.py
def test_svd_compression_accuracy():
    """測試 SVD 壓縮的數值精度"""
    
def test_xkv_prefill_flow():
    """測試 prefill 壓縮流程"""
    
def test_xkv_vs_baseline_ppl():
    """比較壓縮前後的 PPL 差異"""
```

### 2.5 驗收標準
- [ ] SVD 壓縮保持正確的 tensor shape
- [ ] 壓縮後 PPL 增加 < 5% (對於合理的 rank)
- [ ] 無 memory leak

---

## Step 3: 整合 xKV + Paged Attention

### 3.1 目標
- 將壓縮後的 KV 寫回 paged cache
- 支援 decode 階段使用壓縮後的 cache

### 3.2 修改細節

#### 3.2.1 修改 Attention - 添加寫回功能
```python
# layers/attention.py
def write_kvcache(self, k, v, slot_mapping):
    """寫入 KV 到 paged cache"""
    store_kvcache(k, v, self.k_cache, self.v_cache, slot_mapping)
```

#### 3.2.2 修改 xKVCacheManager
```python
# xkv/cache_manager.py
def writeback_compressed_to_paged_cache(self, last_layer_idx, slot_mapping):
    """將壓縮後的 KV 寫回 paged cache"""
    # 遍歷 group 中的所有 layer
    # 調用 attention.write_kvcache()
```

#### 3.2.3 修改 ModelRunner
- 在 prefill 結束後調用 writeback
- 確保 decode 階段能正確使用壓縮後的 cache

### 3.3 測試
```python
# tests/test_xkv_paged.py
def test_paged_writeback_correctness():
    """測試 paged cache 寫回的正確性"""

def test_decode_with_compressed_cache():
    """測試 decode 階段使用壓縮 cache"""

def test_end_to_end_generation():
    """測試完整生成流程"""
```

### 3.4 驗收標準
- [ ] 壓縮後的 KV 能正確寫回 paged cache
- [ ] Decode 階段能正確讀取壓縮後的 cache
- [ ] 生成結果與 Step 2 一致
- [ ] RULER benchmark PPL 差異 < 5%

---

## Step 4: 整合 xKV-sparse

### 4.1 目標
- 結合 xKV 壓縮和 sparse attention (ShadowKV-style)
- 在長序列場景下進一步減少 KV Cache 記憶體

### 4.2 需要創建的檔案
```
nanovllm/
├── xkv/
│   ├── sparse_attention.py  # Sparse attention 實現
│   └── shadowkv.py          # ShadowKV-style 稀疏策略
```

### 4.3 實現細節

#### 4.3.1 Sparse Attention 機制
```python
# xkv/sparse_attention.py
class SparseAttentionSelector:
    """選擇要保留的 KV token"""
    def __init__(self, budget_ratio=0.5):
        self.budget_ratio = budget_ratio

    def select_tokens(self, attention_scores, budget):
        """基於 attention score 選擇重要 token"""
        # 實現 top-k 或 streaming 選擇策略
```

#### 4.3.2 修改 xKVCacheManager
```python
# xkv/cache_manager.py - 添加 sparse 支援
class xKVCacheManager:
    def compress_group_sparse(self, last_layer_idx, attention_scores):
        """先做 SVD 壓縮，再做 sparse 選擇"""
        # 1. SVD 壓縮
        compressed_kv = self.compress_group(last_layer_idx)
        # 2. Sparse 選擇
        sparse_kv = self.sparse_selector.select(compressed_kv, attention_scores)
        return sparse_kv
```

#### 4.3.3 修改 Attention
```python
# layers/attention.py - 添加 sparse 支援
def forward_sparse(self, q, k, v, layer_idx, xkv_manager):
    """支援 sparse attention 的 forward"""
    # 1. 計算 attention scores
    # 2. 選擇重要 token
    # 3. 使用選擇後的 KV 計算 attention
```

### 4.4 測試
```python
# tests/test_xkv_sparse.py
def test_sparse_selection():
    """測試 sparse token 選擇"""

def test_sparse_attention_accuracy():
    """測試 sparse attention 精度"""

def test_xkv_sparse_vs_baseline():
    """比較 xKV-sparse 與 baseline 的 PPL"""

def test_memory_reduction():
    """驗證記憶體節省"""
```

### 4.5 驗收標準
- [ ] Sparse 選擇正確保留重要 token
- [ ] PPL 增加 < 10% (對於 50% sparsity)
- [ ] 記憶體節省符合預期
- [ ] 支援與 CUDA Graph 配合使用

---

## 總結：Code Review Checklist

### Coding Style
- [ ] 遵循現有 nano-vllm 代碼風格
- [ ] 適當的類型標注
- [ ] 清晰的文檔字符串
- [ ] 無冗餘導入

### Inference Efficiency
- [ ] 無不必要的 tensor 複製
- [ ] 適當使用 `torch.no_grad()`
- [ ] CUDA 同步點最小化
- [ ] Memory 高效使用

### Code Elegance
- [ ] 單一職責原則
- [ ] 適當的抽象層次
- [ ] 可讀性高的變量命名
- [ ] 無重複代碼

### 每個 Step 完成後的檢查
1. 運行所有相關測試
2. 檢查 PPL 是否符合預期
3. Review 代碼風格和效率
4. 確認無 memory leak
5. 更新文檔（如需要）

