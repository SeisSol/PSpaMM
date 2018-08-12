
void gemm_sparse (const double* A, const double* B, double* C) {
  __asm__ __volatile__(
    "ldr x0, %0\n\t"
    "ldr x1, %1\n\t"
    "ldr x2, %2\n\t"
      // unrolled_8x56x56
        // for x12 <- 0:1:1)
        "mov x12, #0\r\n"
        "LOOP_TOP_0_%=:\r\n"
          // Unrolling over bn and bk
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "ldr q5, [x1, 0]\r\n"                                       // B[0,0][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,0][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,0][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,0][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,0][0,1]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "ldr q5, [x1, 16]\r\n"                                      // B[0,1][0,1]
            "ldr q7, [x1, 40]\r\n"                                      // B[0,1][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,1][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,1][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,1][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,1][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,1][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,1][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,1][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,1][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "ldr q4, [x1, 8]\r\n"                                       // B[1,1][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[1,1][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[1,1][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[1,1][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[1,1][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "ldr q5, [x1, 24]\r\n"                                      // B[2,1][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[2,1][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[2,1][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[2,1][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[2,1][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "ldr q5, [x1, 32]\r\n"                                      // B[3,1][0,1]
            "ldr q7, [x1, 48]\r\n"                                      // B[3,1][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,1][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,1][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,1][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,1][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,1][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,1][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,1][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,1][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "ldr q6, [x1, 56]\r\n"                                      // B[0,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[0,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[0,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[0,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "ldr q7, [x1, 112]\r\n"                                     // B[1,2][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[1,2][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[1,2][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[1,2][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[1,2][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "ldr q6, [x1, 64]\r\n"                                      // B[2,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[2,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[2,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[2,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "ldr q6, [x1, 72]\r\n"                                      // B[3,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[3,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[3,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[3,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "ldr q6, [x1, 80]\r\n"                                      // B[4,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[4,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[4,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[4,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[4,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "ldr q7, [x1, 120]\r\n"                                     // B[5,2][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[5,2][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[5,2][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[5,2][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[5,2][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "ldr q6, [x1, 88]\r\n"                                      // B[6,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[6,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[6,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[6,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[6,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "ldr q7, [x1, 128]\r\n"                                     // B[7,2][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[7,2][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[7,2][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[7,2][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[7,2][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "ldr q6, [x1, 96]\r\n"                                      // B[8,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[8,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[8,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[8,2][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "ldr q6, [x1, 104]\r\n"                                     // B[9,2][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,2][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[9,2][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[9,2][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[9,2][0,2]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "ldr q4, [x1, 136]\r\n"                                     // B[0,3][0,0]
            "ldr q7, [x1, 200]\r\n"                                     // B[0,3][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[0,3][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,3][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[0,3][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,3][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[0,3][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,3][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[0,3][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,3][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "ldr q6, [x1, 184]\r\n"                                     // B[1,3][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[1,3][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[1,3][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[1,3][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[1,3][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "ldr q4, [x1, 144]\r\n"                                     // B[2,3][0,0]
            "ldr q7, [x1, 208]\r\n"                                     // B[2,3][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[2,3][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,3][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[2,3][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,3][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[2,3][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,3][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[2,3][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,3][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "ldr q4, [x1, 152]\r\n"                                     // B[3,3][0,0]
            "ldr q7, [x1, 216]\r\n"                                     // B[3,3][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[3,3][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,3][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[3,3][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,3][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[3,3][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,3][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[3,3][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,3][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "ldr q4, [x1, 160]\r\n"                                     // B[6,3][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[6,3][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[6,3][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[6,3][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[6,3][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "ldr q6, [x1, 192]\r\n"                                     // B[7,3][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[7,3][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[7,3][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[7,3][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[7,3][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "ldr q4, [x1, 168]\r\n"                                     // B[8,3][0,0]
            "ldr q7, [x1, 224]\r\n"                                     // B[8,3][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[8,3][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,3][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[8,3][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,3][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[8,3][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,3][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[8,3][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,3][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "ldr q4, [x1, 176]\r\n"                                     // B[9,3][0,0]
            "ldr q7, [x1, 232]\r\n"                                     // B[9,3][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[9,3][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,3][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[9,3][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,3][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[9,3][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,3][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[9,3][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,3][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "ldr q5, [x1, 240]\r\n"                                     // B[0,4][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,4][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,4][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,4][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,4][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "ldr q5, [x1, 248]\r\n"                                     // B[3,4][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,4][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,4][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,4][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,4][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #256\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,4][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,4][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[9,4][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[9,4][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[9,4][0,1]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #320\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[0,5][0,1]
            "ldr q7, [x11, 152]\r\n"                                    // B[0,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #264\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[1,5][0,0]
            "ldr q6, [x11, 160]\r\n"                                    // B[1,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[1,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[1,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[1,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[1,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[1,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[1,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[1,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[1,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #328\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[2,5][0,1]
            "ldr q7, [x11, 152]\r\n"                                    // B[2,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[2,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[2,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[2,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[2,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #336\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[3,5][0,1]
            "ldr q7, [x11, 152]\r\n"                                    // B[3,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #344\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[4,5][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[4,5][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[4,5][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[4,5][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[4,5][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "add x11, x1, #272\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[5,5][0,0]
            "ldr q6, [x11, 160]\r\n"                                    // B[5,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[5,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[5,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[5,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[5,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[5,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[5,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[5,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[5,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #352\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[6,5][0,1]
            "ldr q7, [x11, 144]\r\n"                                    // B[6,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[6,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[6,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[6,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[6,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[6,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[6,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[6,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[6,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #280\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[7,5][0,0]
            "ldr q6, [x11, 160]\r\n"                                    // B[7,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[7,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[7,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[7,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[7,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[7,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[7,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[7,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[7,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #360\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[8,5][0,1]
            "ldr q7, [x11, 144]\r\n"                                    // B[8,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[8,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[8,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[8,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[8,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #368\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,5][0,1]
            "ldr q7, [x11, 144]\r\n"                                    // B[9,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[9,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[9,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[9,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "add x11, x0, #640\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,10] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,10] [4,0]
            "add x11, x1, #288\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[10,5][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[10,5][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[10,5][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[10,5][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[10,5][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,11] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,11] [4,0]
            "add x11, x1, #376\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[11,5][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[11,5][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[11,5][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[11,5][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[11,5][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,12] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,12] [4,0]
            "add x11, x1, #296\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[12,5][0,0]
            "ldr q6, [x11, 152]\r\n"                                    // B[12,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[12,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[12,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[12,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[12,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[12,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[12,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[12,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[12,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,13] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,13] [4,0]
            "add x11, x1, #384\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[13,5][0,1]
            "ldr q7, [x11, 136]\r\n"                                    // B[13,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[13,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[13,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[13,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[13,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[13,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[13,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[13,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[13,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #392\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[14,5][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[14,5][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[14,5][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[14,5][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[14,5][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,15] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,15] [4,0]
            "add x11, x1, #304\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[15,5][0,0]
            "ldr q6, [x11, 152]\r\n"                                    // B[15,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[15,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[15,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[15,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[15,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[15,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[15,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[15,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[15,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #400\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[16,5][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[16,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[16,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[16,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[16,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[16,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[16,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[16,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[16,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[16,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #312\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[17,5][0,0]
            "ldr q6, [x11, 152]\r\n"                                    // B[17,5][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[17,5][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[17,5][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[17,5][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[17,5][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[17,5][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[17,5][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[17,5][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[17,5][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #408\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[18,5][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[18,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[18,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[18,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[18,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[18,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[18,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[18,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[18,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[18,5][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #416\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[19,5][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[19,5][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[19,5][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[19,5][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[19,5][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[19,5][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[19,5][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[19,5][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[19,5][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[19,5][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #552\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[0,6][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[0,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #640\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[1,6][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[1,6][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[1,6][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[1,6][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[1,6][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #560\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[2,6][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[2,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[2,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[2,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[2,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[2,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #568\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[3,6][0,1]
            "ldr q7, [x11, 128]\r\n"                                    // B[3,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #576\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[4,6][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[4,6][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[4,6][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[4,6][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[4,6][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "add x11, x1, #648\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[5,6][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[5,6][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[5,6][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[5,6][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[5,6][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #584\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[6,6][0,1]
            "ldr q7, [x11, 120]\r\n"                                    // B[6,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[6,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[6,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[6,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[6,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[6,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[6,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[6,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[6,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #656\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[7,6][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[7,6][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[7,6][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[7,6][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[7,6][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #592\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[8,6][0,1]
            "ldr q7, [x11, 120]\r\n"                                    // B[8,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[8,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[8,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[8,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[8,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #600\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,6][0,1]
            "ldr q7, [x11, 120]\r\n"                                    // B[9,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[9,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[9,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[9,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #608\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[14,6][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[14,6][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[14,6][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[14,6][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[14,6][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,15] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,15] [4,0]
            "add x11, x1, #664\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[15,6][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[15,6][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[15,6][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[15,6][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[15,6][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #616\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[16,6][0,1]
            "ldr q7, [x11, 112]\r\n"                                    // B[16,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[16,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[16,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[16,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[16,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[16,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[16,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[16,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[16,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #672\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[17,6][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[17,6][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[17,6][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[17,6][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[17,6][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #624\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[18,6][0,1]
            "ldr q7, [x11, 112]\r\n"                                    // B[18,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[18,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[18,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[18,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[18,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[18,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[18,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[18,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[18,6][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #632\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[19,6][0,1]
            "ldr q7, [x11, 112]\r\n"                                    // B[19,6][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[19,6][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[19,6][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[19,6][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[19,6][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[19,6][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[19,6][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[19,6][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[19,6][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #776\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[0,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[0,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[0,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[0,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #752\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[1,7][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[1,7][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[1,7][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[1,7][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[1,7][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #784\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[2,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[2,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[2,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[2,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #792\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[3,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[3,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[3,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[3,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #760\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[7,7][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[7,7][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[7,7][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[7,7][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[7,7][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #800\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[8,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[8,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[8,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[8,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #808\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[9,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[9,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[9,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[9,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #768\r\n"                                     // 
            "ldr q5, [x11, 0]\r\n"                                      // B[17,7][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[17,7][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[17,7][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[17,7][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[17,7][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #816\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[18,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[18,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[18,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[18,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[18,7][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #824\r\n"                                     // 
            "ldr q6, [x11, 0]\r\n"                                      // B[19,7][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[19,7][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[19,7][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[19,7][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[19,7][0,2]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #832\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[0,8][0,0]
            "ldr q7, [x11, 32]\r\n"                                     // B[0,8][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[0,8][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,8][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[0,8][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,8][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[0,8][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,8][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[0,8][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #872\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[2,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #840\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[3,8][0,0]
            "ldr q7, [x11, 40]\r\n"                                     // B[3,8][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[3,8][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,8][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[3,8][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,8][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[3,8][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,8][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[3,8][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #888\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[4,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[4,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[4,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[4,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[4,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #896\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[6,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[6,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[6,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[6,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[6,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #904\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[8,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #848\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[9,8][0,0]
            "ldr q7, [x11, 64]\r\n"                                     // B[9,8][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[9,8][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,8][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[9,8][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,8][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[9,8][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,8][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[9,8][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,11] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,11] [4,0]
            "add x11, x1, #920\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[11,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[11,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[11,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[11,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[11,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,13] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,13] [4,0]
            "add x11, x1, #928\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[13,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[13,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[13,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[13,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[13,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #936\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[14,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[14,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[14,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[14,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[14,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #944\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[16,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[16,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[16,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[16,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[16,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #952\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[18,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[18,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[18,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[18,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[18,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #856\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[19,8][0,0]
            "ldr q7, [x11, 104]\r\n"                                    // B[19,8][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[19,8][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[19,8][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[19,8][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[19,8][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[19,8][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[19,8][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[19,8][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[19,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=20)
              "add x11, x0, #1280\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,20] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,20] [4,0]
            "add x11, x1, #968\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[20,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[20,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[20,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[20,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[20,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=22)
              "add x11, x0, #1408\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,22] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,22] [4,0]
            "add x11, x1, #976\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[22,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[22,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[22,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[22,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[22,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=24)
              "add x11, x0, #1536\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,24] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,24] [4,0]
            "add x11, x1, #984\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[24,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[24,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[24,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[24,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[24,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "add x11, x0, #1664\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,26] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,26] [4,0]
            "add x11, x1, #992\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[26,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[26,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[26,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[26,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[26,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,28] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,28] [4,0]
            "add x11, x1, #1000\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[28,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[28,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[28,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[28,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[28,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,29] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,29] [4,0]
            "add x11, x1, #1008\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[29,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[29,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[29,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[29,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[29,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,31] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,31] [4,0]
            "add x11, x1, #1016\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[31,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[31,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[31,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[31,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[31,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,33] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,33] [4,0]
            "add x11, x1, #1024\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[33,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[33,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[33,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[33,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[33,8][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #1032\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[34,8][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[34,8][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[34,8][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[34,8][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[34,8][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #1144\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[0,9][0,1]
            "ldr q7, [x11, 248]\r\n"                                    // B[0,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #1040\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[1,9][0,0]
            "add x11, x1, #1312\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[1,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[1,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[1,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[1,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[1,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[1,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[1,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[1,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[1,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #1152\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[2,9][0,1]
            "ldr q7, [x11, 248]\r\n"                                    // B[2,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[2,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[2,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[2,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[2,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #1160\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[3,9][0,1]
            "ldr q7, [x11, 248]\r\n"                                    // B[3,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #1168\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[4,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[4,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[4,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[4,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[4,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "add x11, x1, #1048\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[5,9][0,0]
            "add x11, x1, #1320\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[5,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[5,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[5,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[5,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[5,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[5,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[5,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[5,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[5,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #1176\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[6,9][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[6,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[6,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[6,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[6,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[6,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[6,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[6,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[6,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[6,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #1056\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[7,9][0,0]
            "add x11, x1, #1328\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[7,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[7,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[7,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[7,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[7,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[7,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[7,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[7,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[7,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #1184\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[8,9][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[8,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[8,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[8,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[8,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[8,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #1192\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,9][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[9,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[9,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[9,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[9,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "add x11, x0, #640\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,10] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,10] [4,0]
            "add x11, x1, #1064\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[10,9][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[10,9][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[10,9][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[10,9][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[10,9][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,11] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,11] [4,0]
            "add x11, x1, #1200\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[11,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[11,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[11,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[11,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[11,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,12] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,12] [4,0]
            "add x11, x1, #1072\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[12,9][0,0]
            "add x11, x1, #1336\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[12,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[12,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[12,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[12,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[12,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[12,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[12,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[12,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[12,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,13] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,13] [4,0]
            "add x11, x1, #1208\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[13,9][0,1]
            "ldr q7, [x11, 232]\r\n"                                    // B[13,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[13,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[13,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[13,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[13,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[13,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[13,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[13,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[13,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #1216\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[14,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[14,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[14,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[14,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[14,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,15] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,15] [4,0]
            "add x11, x1, #1080\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[15,9][0,0]
            "add x11, x1, #1344\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[15,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[15,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[15,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[15,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[15,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[15,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[15,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[15,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[15,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #1224\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[16,9][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[16,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[16,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[16,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[16,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[16,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[16,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[16,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[16,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[16,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #1088\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[17,9][0,0]
            "add x11, x1, #1352\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[17,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[17,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[17,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[17,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[17,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[17,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[17,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[17,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[17,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #1232\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[18,9][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[18,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[18,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[18,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[18,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[18,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[18,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[18,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[18,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[18,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #1240\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[19,9][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[19,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[19,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[19,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[19,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[19,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[19,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[19,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[19,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[19,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=21)
              "add x11, x0, #1344\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,21] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,21] [4,0]
            "add x11, x1, #1096\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[21,9][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[21,9][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[21,9][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[21,9][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[21,9][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=22)
              "add x11, x0, #1408\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,22] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,22] [4,0]
            "add x11, x1, #1248\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[22,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[22,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[22,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[22,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[22,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=23)
              "add x11, x0, #1472\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,23] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,23] [4,0]
            "add x11, x1, #1104\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[23,9][0,0]
            "add x11, x1, #1360\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[23,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[23,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[23,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[23,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[23,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[23,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[23,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[23,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[23,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=24)
              "add x11, x0, #1536\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,24] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,24] [4,0]
            "add x11, x1, #1256\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[24,9][0,1]
            "ldr q7, [x11, 216]\r\n"                                    // B[24,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[24,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[24,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[24,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[24,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[24,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[24,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[24,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[24,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=25)
              "add x11, x0, #1600\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,25] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,25] [4,0]
            "add x11, x1, #1112\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[25,9][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[25,9][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[25,9][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[25,9][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[25,9][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "add x11, x0, #1664\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,26] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,26] [4,0]
            "add x11, x1, #1264\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[26,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[26,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[26,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[26,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[26,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "add x11, x0, #1728\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,27] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,27] [4,0]
            "add x11, x1, #1120\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[27,9][0,0]
            "ldr q6, [x11, 248]\r\n"                                    // B[27,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[27,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[27,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[27,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[27,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[27,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[27,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[27,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[27,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,28] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,28] [4,0]
            "add x11, x1, #1272\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[28,9][0,1]
            "ldr q7, [x11, 208]\r\n"                                    // B[28,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[28,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[28,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[28,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[28,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[28,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[28,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[28,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[28,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,29] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,29] [4,0]
            "add x11, x1, #1280\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[29,9][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[29,9][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[29,9][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[29,9][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[29,9][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "add x11, x0, #1920\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,30] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,30] [4,0]
            "add x11, x1, #1128\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[30,9][0,0]
            "ldr q6, [x11, 248]\r\n"                                    // B[30,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[30,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[30,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[30,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[30,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[30,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[30,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[30,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[30,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,31] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,31] [4,0]
            "add x11, x1, #1288\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[31,9][0,1]
            "ldr q7, [x11, 200]\r\n"                                    // B[31,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[31,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[31,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[31,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[31,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[31,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[31,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[31,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[31,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,32] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,32] [4,0]
            "add x11, x1, #1136\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[32,9][0,0]
            "ldr q6, [x11, 248]\r\n"                                    // B[32,9][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[32,9][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[32,9][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[32,9][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[32,9][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[32,9][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[32,9][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[32,9][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[32,9][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,33] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,33] [4,0]
            "add x11, x1, #1296\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[33,9][0,1]
            "ldr q7, [x11, 200]\r\n"                                    // B[33,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[33,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[33,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[33,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[33,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[33,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[33,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[33,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[33,9][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #1304\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[34,9][0,1]
            "ldr q7, [x11, 200]\r\n"                                    // B[34,9][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[34,9][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[34,9][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[34,9][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[34,9][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[34,9][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[34,9][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[34,9][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[34,9][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #1600\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[0,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[0,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[0,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[0,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #1512\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[1,10][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[1,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[1,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[1,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[1,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[1,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[1,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[1,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[1,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[1,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #1608\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[2,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[2,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[2,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[2,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #1616\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[3,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[3,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[3,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[3,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #1624\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[4,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[4,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[4,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[4,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[4,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "add x11, x1, #1520\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[5,10][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[5,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[5,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[5,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[5,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[5,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[5,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[5,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[5,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[5,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #1632\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[6,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[6,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[6,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[6,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[6,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #1528\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[7,10][0,1]
            "ldr q7, [x11, 240]\r\n"                                    // B[7,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[7,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[7,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[7,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[7,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[7,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[7,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[7,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[7,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #1640\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[8,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[8,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[8,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[8,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #1648\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[9,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[9,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[9,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[9,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "add x11, x0, #640\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,10] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,10] [4,0]
            "add x11, x1, #1536\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[10,10][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[10,10][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[10,10][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[10,10][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[10,10][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,11] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,11] [4,0]
            "add x11, x1, #1656\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[11,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[11,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[11,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[11,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[11,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,12] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,12] [4,0]
            "add x11, x1, #1544\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[12,10][0,1]
            "ldr q7, [x11, 232]\r\n"                                    // B[12,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[12,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[12,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[12,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[12,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[12,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[12,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[12,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[12,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,13] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,13] [4,0]
            "add x11, x1, #1664\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[13,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[13,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[13,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[13,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[13,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #1672\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[14,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[14,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[14,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[14,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[14,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,15] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,15] [4,0]
            "add x11, x1, #1552\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[15,10][0,1]
            "ldr q7, [x11, 232]\r\n"                                    // B[15,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[15,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[15,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[15,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[15,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[15,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[15,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[15,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[15,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #1680\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[16,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[16,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[16,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[16,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[16,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #1560\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[17,10][0,1]
            "ldr q7, [x11, 232]\r\n"                                    // B[17,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[17,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[17,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[17,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[17,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[17,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[17,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[17,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[17,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #1688\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[18,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[18,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[18,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[18,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[18,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #1696\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[19,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[19,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[19,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[19,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[19,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=25)
              "add x11, x0, #1600\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,25] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,25] [4,0]
            "add x11, x1, #1568\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[25,10][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[25,10][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[25,10][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[25,10][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[25,10][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "add x11, x0, #1664\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,26] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,26] [4,0]
            "add x11, x1, #1704\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[26,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[26,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[26,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[26,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[26,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "add x11, x0, #1728\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,27] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,27] [4,0]
            "add x11, x1, #1576\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[27,10][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[27,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[27,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[27,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[27,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[27,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[27,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[27,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[27,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[27,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,28] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,28] [4,0]
            "add x11, x1, #1712\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[28,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[28,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[28,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[28,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[28,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,29] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,29] [4,0]
            "add x11, x1, #1720\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[29,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[29,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[29,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[29,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[29,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "add x11, x0, #1920\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,30] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,30] [4,0]
            "add x11, x1, #1584\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[30,10][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[30,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[30,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[30,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[30,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[30,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[30,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[30,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[30,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[30,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,31] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,31] [4,0]
            "add x11, x1, #1728\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[31,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[31,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[31,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[31,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[31,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,32] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,32] [4,0]
            "add x11, x1, #1592\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[32,10][0,1]
            "ldr q7, [x11, 224]\r\n"                                    // B[32,10][0,3]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[32,10][0,1]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[32,10][0,3]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[32,10][0,1]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[32,10][0,3]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[32,10][0,1]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[32,10][0,3]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[32,10][0,1]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[32,10][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,33] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,33] [4,0]
            "add x11, x1, #1736\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[33,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[33,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[33,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[33,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[33,10][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #1744\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[34,10][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[34,10][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[34,10][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[34,10][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[34,10][0,2]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #1824\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[0,11][0,0]
            "ldr q6, [x11, 112]\r\n"                                    // B[0,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[0,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[0,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[0,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[0,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[0,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[0,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[0,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #2056\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[1,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[1,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[1,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[1,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[1,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #1832\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[2,11][0,0]
            "ldr q6, [x11, 112]\r\n"                                    // B[2,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[2,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[2,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[2,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[2,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[2,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[2,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[2,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #1840\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[3,11][0,0]
            "ldr q6, [x11, 112]\r\n"                                    // B[3,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[3,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[3,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[3,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[3,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[3,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[3,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[3,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,4] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,4] [4,0]
            "add x11, x1, #1960\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[4,11][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[4,11][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[4,11][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[4,11][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[4,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,5] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,5] [4,0]
            "add x11, x1, #2064\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[5,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[5,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[5,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[5,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[5,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #1848\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[6,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[6,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[6,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[6,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[6,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[6,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[6,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[6,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[6,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[6,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #2072\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[7,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[7,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[7,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[7,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[7,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #1856\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[8,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[8,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[8,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[8,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[8,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[8,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[8,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[8,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[8,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #1864\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[9,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[9,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[9,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[9,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[9,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[9,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[9,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[9,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[9,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,13] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,13] [4,0]
            "add x11, x1, #1872\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[13,11][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[13,11][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[13,11][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[13,11][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[13,11][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,14] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,14] [4,0]
            "add x11, x1, #1992\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[14,11][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[14,11][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[14,11][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[14,11][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[14,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,15] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,15] [4,0]
            "add x11, x1, #2080\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[15,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[15,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[15,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[15,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[15,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #1880\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[16,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[16,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[16,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[16,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[16,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[16,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[16,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[16,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[16,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[16,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #2088\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[17,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[17,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[17,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[17,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[17,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #1888\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[18,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[18,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[18,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[18,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[18,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[18,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[18,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[18,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[18,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[18,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #1896\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[19,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[19,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[19,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[19,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[19,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[19,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[19,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[19,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[19,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[19,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,28] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,28] [4,0]
            "add x11, x1, #1904\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[28,11][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[28,11][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[28,11][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[28,11][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[28,11][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,29] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,29] [4,0]
            "add x11, x1, #2024\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[29,11][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[29,11][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[29,11][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[29,11][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[29,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "add x11, x0, #1920\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,30] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,30] [4,0]
            "add x11, x1, #2096\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[30,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[30,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[30,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[30,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[30,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,31] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,31] [4,0]
            "add x11, x1, #1912\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[31,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[31,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[31,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[31,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[31,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[31,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[31,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[31,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[31,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[31,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,32] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,32] [4,0]
            "add x11, x1, #2104\r\n"                                    // 
            "ldr q7, [x11, 0]\r\n"                                      // B[32,11][0,3]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[32,11][0,3]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[32,11][0,3]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[32,11][0,3]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[32,11][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,33] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,33] [4,0]
            "add x11, x1, #1920\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[33,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[33,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[33,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[33,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[33,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[33,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[33,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[33,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[33,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[33,11][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #1928\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[34,11][0,0]
            "ldr q6, [x11, 120]\r\n"                                    // B[34,11][0,2]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[34,11][0,0]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[34,11][0,2]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[34,11][0,0]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[34,11][0,2]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[34,11][0,0]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[34,11][0,2]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[34,11][0,0]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[34,11][0,2]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #2112\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[0,12][0,0]
            "ldr q7, [x11, 128]\r\n"                                    // B[0,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[0,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[0,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[0,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[0,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[0,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[0,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[0,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldp q0, q1, [x0, 64]\r\n"                                  // A [0,1] [0,0]
              "ldp q2, q3, [x0, 96]\r\n"                                  // A [0,1] [4,0]
            "add x11, x1, #2208\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[1,12][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[1,12][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[1,12][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[1,12][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[1,12][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldp q0, q1, [x0, 128]\r\n"                                 // A [0,2] [0,0]
              "ldp q2, q3, [x0, 160]\r\n"                                 // A [0,2] [4,0]
            "add x11, x1, #2120\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[2,12][0,0]
            "ldr q7, [x11, 128]\r\n"                                    // B[2,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[2,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[2,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[2,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[2,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[2,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[2,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[2,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[2,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #2128\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[3,12][0,0]
            "ldr q7, [x11, 128]\r\n"                                    // B[3,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[3,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[3,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[3,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[3,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[3,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[3,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[3,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,6] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,6] [4,0]
            "add x11, x1, #2136\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[6,12][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[6,12][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[6,12][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[6,12][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[6,12][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,7] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,7] [4,0]
            "add x11, x1, #2216\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[7,12][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[7,12][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[7,12][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[7,12][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[7,12][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,8] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,8] [4,0]
            "add x11, x1, #2144\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[8,12][0,0]
            "ldr q7, [x11, 120]\r\n"                                    // B[8,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[8,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[8,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[8,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[8,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[8,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[8,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[8,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[8,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #2152\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[9,12][0,0]
            "ldr q7, [x11, 120]\r\n"                                    // B[9,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[9,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[9,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[9,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[9,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[9,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[9,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[9,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,16] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,16] [4,0]
            "add x11, x1, #2160\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[16,12][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[16,12][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[16,12][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[16,12][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[16,12][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,17] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,17] [4,0]
            "add x11, x1, #2224\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[17,12][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[17,12][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[17,12][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[17,12][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[17,12][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,18] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,18] [4,0]
            "add x11, x1, #2168\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[18,12][0,0]
            "ldr q7, [x11, 112]\r\n"                                    // B[18,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[18,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[18,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[18,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[18,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[18,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[18,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[18,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[18,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #2176\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[19,12][0,0]
            "ldr q7, [x11, 112]\r\n"                                    // B[19,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[19,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[19,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[19,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[19,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[19,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[19,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[19,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[19,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,31] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,31] [4,0]
            "add x11, x1, #2184\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[31,12][0,0]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[31,12][0,0]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[31,12][0,0]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[31,12][0,0]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[31,12][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,32] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,32] [4,0]
            "add x11, x1, #2232\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[32,12][0,2]
            "fmla v24.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[32,12][0,2]
            "fmla v25.2d, v1.2d, v6.1d[0]\r\n"                          // C[2:4,2] += A[2:4,0]*B[32,12][0,2]
            "fmla v26.2d, v2.2d, v6.1d[0]\r\n"                          // C[4:6,2] += A[4:6,0]*B[32,12][0,2]
            "fmla v27.2d, v3.2d, v6.1d[0]\r\n"                          // C[6:8,2] += A[6:8,0]*B[32,12][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,33] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,33] [4,0]
            "add x11, x1, #2192\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[33,12][0,0]
            "ldr q7, [x11, 104]\r\n"                                    // B[33,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[33,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[33,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[33,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[33,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[33,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[33,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[33,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[33,12][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #2200\r\n"                                    // 
            "ldr q4, [x11, 0]\r\n"                                      // B[34,12][0,0]
            "ldr q7, [x11, 104]\r\n"                                    // B[34,12][0,3]
            "fmla v16.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[34,12][0,0]
            "fmla v28.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[34,12][0,3]
            "fmla v17.2d, v1.2d, v4.1d[0]\r\n"                          // C[2:4,0] += A[2:4,0]*B[34,12][0,0]
            "fmla v29.2d, v1.2d, v7.1d[0]\r\n"                          // C[2:4,3] += A[2:4,0]*B[34,12][0,3]
            "fmla v18.2d, v2.2d, v4.1d[0]\r\n"                          // C[4:6,0] += A[4:6,0]*B[34,12][0,0]
            "fmla v30.2d, v2.2d, v7.1d[0]\r\n"                          // C[4:6,3] += A[4:6,0]*B[34,12][0,3]
            "fmla v19.2d, v3.2d, v4.1d[0]\r\n"                          // C[6:8,0] += A[6:8,0]*B[34,12][0,0]
            "fmla v31.2d, v3.2d, v7.1d[0]\r\n"                          // C[6:8,3] += A[6:8,0]*B[34,12][0,3]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
          "add x2, x2, #256\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
            "fmov d16, xzr\r\n"
            "fmov d17, xzr\r\n"
            "fmov d18, xzr\r\n"
            "fmov d19, xzr\r\n"
            "fmov d20, xzr\r\n"
            "fmov d21, xzr\r\n"
            "fmov d22, xzr\r\n"
            "fmov d23, xzr\r\n"
            "fmov d24, xzr\r\n"
            "fmov d25, xzr\r\n"
            "fmov d26, xzr\r\n"
            "fmov d27, xzr\r\n"
            "fmov d28, xzr\r\n"
            "fmov d29, xzr\r\n"
            "fmov d30, xzr\r\n"
            "fmov d31, xzr\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "ldp q0, q1, [x0, 0]\r\n"                                   // A [0,0] [0,0]
              "ldp q2, q3, [x0, 32]\r\n"                                  // A [0,0] [4,0]
            "add x11, x1, #2312\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[0,13][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,13][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[0,13][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[0,13][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[0,13][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldp q0, q1, [x0, 192]\r\n"                                 // A [0,3] [0,0]
              "ldp q2, q3, [x0, 224]\r\n"                                 // A [0,3] [4,0]
            "add x11, x1, #2320\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[3,13][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,13][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[3,13][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[3,13][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[3,13][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,9] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,9] [4,0]
            "add x11, x1, #2328\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,13][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,13][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[9,13][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[9,13][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[9,13][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,19] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,19] [4,0]
            "add x11, x1, #2336\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[19,13][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[19,13][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[19,13][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[19,13][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[19,13][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldp q0, q1, [x11, 0]\r\n"                                  // A [0,34] [0,0]
              "ldp q2, q3, [x11, 32]\r\n"                                 // A [0,34] [4,0]
            "add x11, x1, #2344\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[34,13][0,1]
            "fmla v20.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[34,13][0,1]
            "fmla v21.2d, v1.2d, v5.1d[0]\r\n"                          // C[2:4,1] += A[2:4,0]*B[34,13][0,1]
            "fmla v22.2d, v2.2d, v5.1d[0]\r\n"                          // C[4:6,1] += A[4:6,0]*B[34,13][0,1]
            "fmla v23.2d, v3.2d, v5.1d[0]\r\n"                          // C[6:8,1] += A[6:8,0]*B[34,13][0,1]
            // Store C register block @ (d=0,r=0)
            "stp q16, q17, [x2, 0]\r\n"                                 // C [0,0] [0,0]
            "stp q18, q19, [x2, 32]\r\n"                                // C [0,0] [4,0]
            "stp q20, q21, [x2, 64]\r\n"                                // C [0,0] [0,1]
            "stp q22, q23, [x2, 96]\r\n"                                // C [0,0] [4,1]
            "stp q24, q25, [x2, 128]\r\n"                               // C [0,0] [0,2]
            "stp q26, q27, [x2, 160]\r\n"                               // C [0,0] [4,2]
            "stp q28, q29, [x2, 192]\r\n"                               // C [0,0] [0,3]
            "stp q30, q31, [x2, 224]\r\n"                               // C [0,0] [4,3]
        "add x0, x0, #64\r\n"                                       // Move A to (d=1,r=0)
        "add x2, x2, #-3264\r\n"                                    // Move C to (d=1,r=-13)
        "add x12, x12, #1\r\n"
        "cmp x12, #1\r\n"
        "b.lo LOOP_TOP_0_%=\r\n"

    : : "m"(A), "m"(B), "m"(C) : "r0","r11","r12","r2","v0","v16","v17","v18","v19","v2","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","v4","v5","v6","v7");

};
