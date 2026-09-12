# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""BMM FP8 Expert mode test: 3D batch GEMM + Vector scale multiply.

Demonstrates:
  - Tail block handling (remain_M/remain_N/remain_K)
  - 3D data processing ([B, M, K] x [B, K, N])
  - Cube-Vector interaction (T.Scope("Cube") GEMM -> T.Scope("Vector") scale)
  - FP8 simulation via float16 + scale factors
"""

import os
import torch
import tilelang
import tilelang.language as T


def bmm_fp8_exp(block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    B = T.symbolic("B")
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def kernel(
        A: T.Tensor((B, M, K), dtype),
        B_t: T.Tensor((B, K, N), dtype),
        scale_a: T.Tensor((1, 1, 1), "float32"),
        scale_b: T.Tensor((1, 1, 1), "float32"),
        C: T.Tensor((B, M, N), accum_dtype),
    ):
        with T.Kernel(B * m_num * n_num, is_npu=True) as (bid, subid):
            batch = bid // (m_num * n_num)
            tile_id = bid % (m_num * n_num)
            bx = tile_id // n_num * block_M
            by = tile_id % n_num * block_N

            # -- Phase 1: Cube (GEMM) --
            with T.Scope("Cube"):
                remain_M = T.min(M - bx, block_M)
                remain_N = T.min(N - by, block_N)

                A_buf = T.alloc_L1((block_M, block_K), dtype)
                B_buf = T.alloc_L1((block_K, block_N), dtype)
                C_buf = T.alloc_L0C((block_M, block_N), accum_dtype)

                for ki in T.serial(T.ceildiv(K, block_K)):
                    k_start = ki * block_K
                    remain_K = T.min(K - k_start, block_K)

                    T.load_nd2nz(A[batch, bx, k_start], A_buf, [remain_M, remain_K])
                    T.load_nd2nz(B_t[batch, k_start, by], B_buf, [remain_K, remain_N])
                    T.gemm(
                        A_buf,
                        B_buf,
                        C_buf,
                        initC=(ki == 0),
                        b_transpose=False,
                        size=[remain_M, remain_K, remain_N],
                    )

                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(1)
                    T.store_fixpipe(
                        C_buf,
                        C[batch, bx, by],
                        size=[remain_M, remain_N],
                        enable_nz2nd=True,
                    )
                    T.sync_block_set(0)

            # -- Phase 2: Vector (element-wise scale mul) --
            with T.Scope("Vector"):
                remain_M = T.min(M - bx, block_M)
                remain_N = T.min(N - by, block_N)
                C_vec = T.alloc_ub((1, block_M, block_N), accum_dtype)
                S_buf = T.alloc_ub((1, block_M, block_N), accum_dtype)
                scale_ub = T.alloc_ub((1, 1, 1), "float32")

                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_wait(0)
                    T.copy(C[batch, bx:bx+block_M, by:by+block_N], C_vec)
                    T.copy(scale_a, scale_ub)
                    T.sync_block_set(1)

                T.npuir_brc(scale_ub, S_buf)
                T.vmul(C_vec, S_buf, C_vec)
                T.copy(scale_b, scale_ub)
                T.npuir_brc(scale_ub, S_buf)
                T.vmul(C_vec, S_buf, C_vec)

                T.copy(C_vec[0, 0:remain_M, 0:remain_N], C[batch, bx:bx+remain_M, by:by+remain_N])

    return kernel


def gen_fp8_input(shape, device="npu"):
    x = torch.randn(shape, dtype=torch.float32) * 96.0
    x = torch.clamp(x, -448.0, 448.0)
    return x.to(torch.float16).to(device).contiguous()


def test_bmm_fp8():
    B, M, N, K = 2, 400, 128, 512
    block_M, block_N, block_K = 32, 128, 32

    A = gen_fp8_input((B, M, K))
    B_t = gen_fp8_input((B, K, N))
    scale_a = torch.tensor([[[0.95]]], dtype=torch.float32).npu()
    scale_b = torch.tensor([[[1.05]]], dtype=torch.float32).npu()
    C = torch.zeros(B, M, N, dtype=torch.float32).npu()

    kernel = bmm_fp8_exp(block_M, block_N, block_K)
    compiled = tilelang.compile(kernel, target="npuir")
    compiled(A, B_t, scale_a, scale_b, C)

    # Reference: C = A @ B * scale_a * scale_b
    ref = torch.zeros(B, M, N, dtype=torch.float32)
    for b in range(B):
        ref[b] = (A[b].float() @ B_t[b].float()) * scale_a.item() * scale_b.item()

    torch.testing.assert_close(C.cpu(), ref, rtol=1e-2, atol=2e-2)
    print(f"\033[92m[PASS] BMM FP8 Exp (B={B}, M={M}, N={N}, K={K})\033[0m")  


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    # os.environ["USE_NPUIR_STR"] = "true"
    tilelang.cache.clear_cache()
    test_bmm_fp8()
