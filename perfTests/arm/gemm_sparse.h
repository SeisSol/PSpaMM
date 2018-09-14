
void gemm_sparse (const double* A, const double* B, double* C) {
  __asm__ __volatile__(
    "ldr x0, %0\n\t"
    "ldr x1, %1\n\t"
    "ldr x2, %2\n\t"
          // unrolled_8x56x56
        // for x12 <- 0:1:4)
        "mov x12, #0\r\n"
        "LOOP_TOP_0_%=:\r\n"
          // Unrolling over bn and bk
            // zero registers
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
              "ldr q0, [x0, 0]\r\n"                                       // A [0,0] [0,0]
            "ldr q2, [x1, 0]\r\n"                                       // B[0,0][0,1]
            "ldr q6, [x1, 16]\r\n"                                      // B[0,0][0,5]
            "ldr q8, [x1, 40]\r\n"                                      // B[0,0][0,7]
            "ldr q11, [x1, 56]\r\n"                                     // B[0,0][0,10]
            "ldr q13, [x1, 136]\r\n"                                    // B[0,0][0,12]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,0][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[0,0][0,5]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[0,0][0,7]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[0,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[0,0][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldr q0, [x0, 64]\r\n"                                      // A [0,1] [0,0]
            "ldr q5, [x1, 8]\r\n"                                       // B[1,0][0,4]
            "ldr q12, [x1, 112]\r\n"                                    // B[1,0][0,11]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[1,0][0,4]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[1,0][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldr q0, [x0, 128]\r\n"                                     // A [0,2] [0,0]
            "ldr q6, [x1, 24]\r\n"                                      // B[2,0][0,5]
            "ldr q11, [x1, 64]\r\n"                                     // B[2,0][0,10]
            "ldr q13, [x1, 144]\r\n"                                    // B[2,0][0,12]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[2,0][0,5]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[2,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[2,0][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldr q0, [x0, 192]\r\n"                                     // A [0,3] [0,0]
            "ldr q6, [x1, 32]\r\n"                                      // B[3,0][0,5]
            "ldr q8, [x1, 48]\r\n"                                      // B[3,0][0,7]
            "ldr q11, [x1, 72]\r\n"                                     // B[3,0][0,10]
            "ldr q13, [x1, 152]\r\n"                                    // B[3,0][0,12]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[3,0][0,5]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[3,0][0,7]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[3,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[3,0][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,4] [0,0]
            "ldr q11, [x1, 80]\r\n"                                     // B[4,0][0,10]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[4,0][0,10]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,5] [0,0]
            "ldr q12, [x1, 120]\r\n"                                    // B[5,0][0,11]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[5,0][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,6] [0,0]
            "ldr q11, [x1, 88]\r\n"                                     // B[6,0][0,10]
            "ldr q13, [x1, 160]\r\n"                                    // B[6,0][0,12]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[6,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[6,0][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,7] [0,0]
            "ldr q12, [x1, 128]\r\n"                                    // B[7,0][0,11]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[7,0][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,8] [0,0]
            "ldr q11, [x1, 96]\r\n"                                     // B[8,0][0,10]
            "ldr q13, [x1, 168]\r\n"                                    // B[8,0][0,12]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[8,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[8,0][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,9] [0,0]
            "ldr q11, [x1, 104]\r\n"                                    // B[9,0][0,10]
            "ldr q13, [x1, 176]\r\n"                                    // B[9,0][0,12]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[9,0][0,10]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[9,0][0,12]
            // Store C register block @ (d=0,r=0)
            "str q18, [x2, 0]\r\n"                                      // C [0,0] [0,0]
            "str q19, [x2, 64]\r\n"                                     // C [0,0] [0,1]
            "str q20, [x2, 128]\r\n"                                    // C [0,0] [0,2]
            "str q21, [x2, 192]\r\n"                                    // C [0,0] [0,3]
            "add x11, x2, #256\r\n"                                     // 
            "str q22, [x11, 0]\r\n"                                     // C [0,0] [0,4]
            "str q23, [x11, 64]\r\n"                                    // C [0,0] [0,5]
            "str q24, [x11, 128]\r\n"                                   // C [0,0] [0,6]
            "str q25, [x11, 192]\r\n"                                   // C [0,0] [0,7]
            "add x11, x2, #512\r\n"                                     // 
            "str q26, [x11, 0]\r\n"                                     // C [0,0] [0,8]
            "str q27, [x11, 64]\r\n"                                    // C [0,0] [0,9]
            "str q28, [x11, 128]\r\n"                                   // C [0,0] [0,10]
            "str q29, [x11, 192]\r\n"                                   // C [0,0] [0,11]
            "add x11, x2, #768\r\n"                                     // 
            "str q30, [x11, 0]\r\n"                                     // C [0,0] [0,12]
            "str q31, [x11, 64]\r\n"                                    // C [0,0] [0,13]
          "add x2, x2, #896\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
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
              "ldr q0, [x0, 0]\r\n"                                       // A [0,0] [0,0]
            "ldr q2, [x1, 200]\r\n"                                     // B[0,1][0,1]
            "ldr q4, [x1, 240]\r\n"                                     // B[0,1][0,3]
            "add x11, x1, #320\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[0,1][0,7]
            "ldr q10, [x11, 152]\r\n"                                   // B[0,1][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[0,1][0,11]
            "add x11, x1, #680\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[0,1][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[0,1][0,1]
            "fmla v21.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[0,1][0,3]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[0,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[0,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[0,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[0,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldr q0, [x0, 64]\r\n"                                      // A [0,1] [0,0]
            "ldr q1, [x1, 184]\r\n"                                     // B[1,1][0,0]
            "add x11, x1, #264\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[1,1][0,6]
            "ldr q9, [x11, 160]\r\n"                                    // B[1,1][0,8]
            "add x11, x1, #640\r\n"                                     // 
            "ldr q13, [x11, 0]\r\n"                                     // B[1,1][0,12]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[1,1][0,0]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[1,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[1,1][0,8]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[1,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldr q0, [x0, 128]\r\n"                                     // A [0,2] [0,0]
            "ldr q2, [x1, 208]\r\n"                                     // B[2,1][0,1]
            "add x11, x1, #328\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[2,1][0,7]
            "ldr q10, [x11, 152]\r\n"                                   // B[2,1][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[2,1][0,11]
            "add x11, x1, #688\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[2,1][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[2,1][0,1]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[2,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[2,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[2,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[2,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldr q0, [x0, 192]\r\n"                                     // A [0,3] [0,0]
            "ldr q2, [x1, 216]\r\n"                                     // B[3,1][0,1]
            "ldr q4, [x1, 248]\r\n"                                     // B[3,1][0,3]
            "add x11, x1, #336\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[3,1][0,7]
            "ldr q10, [x11, 152]\r\n"                                   // B[3,1][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[3,1][0,11]
            "add x11, x1, #696\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[3,1][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[3,1][0,1]
            "fmla v21.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[3,1][0,3]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[3,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[3,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[3,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[3,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,4] [0,0]
            "add x11, x1, #344\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[4,1][0,7]
            "ldr q12, [x11, 232]\r\n"                                   // B[4,1][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[4,1][0,7]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[4,1][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,5] [0,0]
            "add x11, x1, #272\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[5,1][0,6]
            "ldr q9, [x11, 160]\r\n"                                    // B[5,1][0,8]
            "add x11, x1, #648\r\n"                                     // 
            "ldr q13, [x11, 0]\r\n"                                     // B[5,1][0,12]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[5,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[5,1][0,8]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[5,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,6] [0,0]
            "add x11, x1, #352\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[6,1][0,7]
            "ldr q10, [x11, 144]\r\n"                                   // B[6,1][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[6,1][0,11]
            "add x11, x1, #704\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[6,1][0,13]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[6,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[6,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[6,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[6,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,7] [0,0]
            "ldr q1, [x1, 192]\r\n"                                     // B[7,1][0,0]
            "add x11, x1, #280\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[7,1][0,6]
            "ldr q9, [x11, 160]\r\n"                                    // B[7,1][0,8]
            "add x11, x1, #656\r\n"                                     // 
            "ldr q13, [x11, 0]\r\n"                                     // B[7,1][0,12]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[7,1][0,0]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[7,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[7,1][0,8]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[7,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,8] [0,0]
            "ldr q2, [x1, 224]\r\n"                                     // B[8,1][0,1]
            "add x11, x1, #360\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[8,1][0,7]
            "ldr q10, [x11, 144]\r\n"                                   // B[8,1][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[8,1][0,11]
            "add x11, x1, #712\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[8,1][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[8,1][0,1]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[8,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[8,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[8,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[8,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,9] [0,0]
            "ldr q2, [x1, 232]\r\n"                                     // B[9,1][0,1]
            "add x11, x1, #256\r\n"                                     // 
            "ldr q4, [x11, 0]\r\n"                                      // B[9,1][0,3]
            "ldr q8, [x11, 112]\r\n"                                    // B[9,1][0,7]
            "add x11, x1, #512\r\n"                                     // 
            "ldr q10, [x11, 0]\r\n"                                     // B[9,1][0,9]
            "ldr q12, [x11, 88]\r\n"                                    // B[9,1][0,11]
            "ldr q14, [x11, 208]\r\n"                                   // B[9,1][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[9,1][0,1]
            "fmla v21.2d, v0.2d, v4.1d[0]\r\n"                          // C[0:2,3] += A[0:2,0]*B[9,1][0,3]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[9,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[9,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[9,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[9,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "add x11, x0, #640\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,10] [0,0]
            "add x11, x1, #288\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[10,1][0,6]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[10,1][0,6]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,11] [0,0]
            "add x11, x1, #376\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[11,1][0,7]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[11,1][0,7]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,12] [0,0]
            "add x11, x1, #296\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[12,1][0,6]
            "ldr q9, [x11, 152]\r\n"                                    // B[12,1][0,8]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[12,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[12,1][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,13] [0,0]
            "add x11, x1, #384\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[13,1][0,7]
            "ldr q10, [x11, 136]\r\n"                                   // B[13,1][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[13,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[13,1][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,14] [0,0]
            "add x11, x1, #392\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[14,1][0,7]
            "ldr q12, [x11, 216]\r\n"                                   // B[14,1][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[14,1][0,7]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[14,1][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,15] [0,0]
            "add x11, x1, #304\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[15,1][0,6]
            "ldr q9, [x11, 152]\r\n"                                    // B[15,1][0,8]
            "add x11, x1, #664\r\n"                                     // 
            "ldr q13, [x11, 0]\r\n"                                     // B[15,1][0,12]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[15,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[15,1][0,8]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[15,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,16] [0,0]
            "add x11, x1, #400\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[16,1][0,7]
            "ldr q10, [x11, 128]\r\n"                                   // B[16,1][0,9]
            "ldr q12, [x11, 216]\r\n"                                   // B[16,1][0,11]
            "add x11, x1, #728\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[16,1][0,13]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[16,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[16,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[16,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[16,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,17] [0,0]
            "add x11, x1, #312\r\n"                                     // 
            "ldr q7, [x11, 0]\r\n"                                      // B[17,1][0,6]
            "ldr q9, [x11, 152]\r\n"                                    // B[17,1][0,8]
            "add x11, x1, #672\r\n"                                     // 
            "ldr q13, [x11, 0]\r\n"                                     // B[17,1][0,12]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[17,1][0,6]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[17,1][0,8]
            "fmla v30.2d, v0.2d, v13.1d[0]\r\n"                         // C[0:2,12] += A[0:2,0]*B[17,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,18] [0,0]
            "add x11, x1, #408\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[18,1][0,7]
            "ldr q10, [x11, 128]\r\n"                                   // B[18,1][0,9]
            "ldr q12, [x11, 216]\r\n"                                   // B[18,1][0,11]
            "add x11, x1, #736\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[18,1][0,13]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[18,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[18,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[18,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[18,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,19] [0,0]
            "add x11, x1, #416\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[19,1][0,7]
            "ldr q10, [x11, 128]\r\n"                                   // B[19,1][0,9]
            "ldr q12, [x11, 216]\r\n"                                   // B[19,1][0,11]
            "add x11, x1, #744\r\n"                                     // 
            "ldr q14, [x11, 0]\r\n"                                     // B[19,1][0,13]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[19,1][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[19,1][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[19,1][0,11]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[19,1][0,13]
            // Store C register block @ (d=0,r=0)
            "str q18, [x2, 0]\r\n"                                      // C [0,0] [0,0]
            "str q19, [x2, 64]\r\n"                                     // C [0,0] [0,1]
            "str q20, [x2, 128]\r\n"                                    // C [0,0] [0,2]
            "str q21, [x2, 192]\r\n"                                    // C [0,0] [0,3]
            "add x11, x2, #256\r\n"                                     // 
            "str q22, [x11, 0]\r\n"                                     // C [0,0] [0,4]
            "str q23, [x11, 64]\r\n"                                    // C [0,0] [0,5]
            "str q24, [x11, 128]\r\n"                                   // C [0,0] [0,6]
            "str q25, [x11, 192]\r\n"                                   // C [0,0] [0,7]
            "add x11, x2, #512\r\n"                                     // 
            "str q26, [x11, 0]\r\n"                                     // C [0,0] [0,8]
            "str q27, [x11, 64]\r\n"                                    // C [0,0] [0,9]
            "str q28, [x11, 128]\r\n"                                   // C [0,0] [0,10]
            "str q29, [x11, 192]\r\n"                                   // C [0,0] [0,11]
            "add x11, x2, #768\r\n"                                     // 
            "str q30, [x11, 0]\r\n"                                     // C [0,0] [0,12]
            "str q31, [x11, 64]\r\n"                                    // C [0,0] [0,13]
          "add x2, x2, #896\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
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
              "ldr q0, [x0, 0]\r\n"                                       // A [0,0] [0,0]
            "add x11, x1, #776\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[0,2][0,2]
            "ldr q5, [x11, 56]\r\n"                                     // B[0,2][0,4]
            "ldr q8, [x11, 88]\r\n"                                     // B[0,2][0,7]
            "add x11, x1, #1144\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[0,2][0,9]
            "ldr q12, [x11, 248]\r\n"                                   // B[0,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,2][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[0,2][0,4]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[0,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[0,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[0,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldr q0, [x0, 64]\r\n"                                      // A [0,1] [0,0]
            "add x11, x1, #752\r\n"                                     // 
            "ldr q2, [x11, 0]\r\n"                                      // B[1,2][0,1]
            "add x11, x1, #1040\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[1,2][0,8]
            "add x11, x1, #1312\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[1,2][0,10]
            "ldr q14, [x11, 200]\r\n"                                   // B[1,2][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[1,2][0,1]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[1,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[1,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[1,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldr q0, [x0, 128]\r\n"                                     // A [0,2] [0,0]
            "add x11, x1, #784\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[2,2][0,2]
            "ldr q8, [x11, 88]\r\n"                                     // B[2,2][0,7]
            "add x11, x1, #1152\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[2,2][0,9]
            "ldr q12, [x11, 248]\r\n"                                   // B[2,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,2][0,2]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[2,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[2,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[2,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldr q0, [x0, 192]\r\n"                                     // A [0,3] [0,0]
            "add x11, x1, #792\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[3,2][0,2]
            "ldr q5, [x11, 48]\r\n"                                     // B[3,2][0,4]
            "ldr q8, [x11, 88]\r\n"                                     // B[3,2][0,7]
            "add x11, x1, #1160\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[3,2][0,9]
            "ldr q12, [x11, 248]\r\n"                                   // B[3,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,2][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[3,2][0,4]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[3,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[3,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[3,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,4] [0,0]
            "add x11, x1, #888\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[4,2][0,7]
            "add x11, x1, #1168\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[4,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[4,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[4,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,5] [0,0]
            "add x11, x1, #1048\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[5,2][0,8]
            "add x11, x1, #1320\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[5,2][0,10]
            "ldr q14, [x11, 200]\r\n"                                   // B[5,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[5,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[5,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[5,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,6] [0,0]
            "add x11, x1, #896\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[6,2][0,7]
            "add x11, x1, #1176\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[6,2][0,9]
            "ldr q12, [x11, 240]\r\n"                                   // B[6,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[6,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[6,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[6,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,7] [0,0]
            "add x11, x1, #760\r\n"                                     // 
            "ldr q2, [x11, 0]\r\n"                                      // B[7,2][0,1]
            "add x11, x1, #1056\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[7,2][0,8]
            "add x11, x1, #1328\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[7,2][0,10]
            "ldr q14, [x11, 200]\r\n"                                   // B[7,2][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[7,2][0,1]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[7,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[7,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[7,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,8] [0,0]
            "add x11, x1, #800\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[8,2][0,2]
            "ldr q8, [x11, 104]\r\n"                                    // B[8,2][0,7]
            "add x11, x1, #1184\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[8,2][0,9]
            "ldr q12, [x11, 240]\r\n"                                   // B[8,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,2][0,2]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[8,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[8,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[8,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,9] [0,0]
            "add x11, x1, #808\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[9,2][0,2]
            "ldr q5, [x11, 40]\r\n"                                     // B[9,2][0,4]
            "ldr q8, [x11, 104]\r\n"                                    // B[9,2][0,7]
            "add x11, x1, #1192\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[9,2][0,9]
            "ldr q12, [x11, 240]\r\n"                                   // B[9,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,2][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[9,2][0,4]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[9,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[9,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[9,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "add x11, x0, #640\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,10] [0,0]
            "add x11, x1, #1064\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[10,2][0,8]
            "add x11, x1, #1536\r\n"                                    // 
            "ldr q14, [x11, 0]\r\n"                                     // B[10,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[10,2][0,8]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[10,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,11] [0,0]
            "add x11, x1, #920\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[11,2][0,7]
            "add x11, x1, #1200\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[11,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[11,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[11,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,12] [0,0]
            "add x11, x1, #1072\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[12,2][0,8]
            "add x11, x1, #1336\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[12,2][0,10]
            "ldr q14, [x11, 208]\r\n"                                   // B[12,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[12,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[12,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[12,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,13] [0,0]
            "add x11, x1, #928\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[13,2][0,7]
            "add x11, x1, #1208\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[13,2][0,9]
            "ldr q12, [x11, 232]\r\n"                                   // B[13,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[13,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[13,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[13,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,14] [0,0]
            "add x11, x1, #936\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[14,2][0,7]
            "add x11, x1, #1216\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[14,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[14,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[14,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,15] [0,0]
            "add x11, x1, #1080\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[15,2][0,8]
            "add x11, x1, #1344\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[15,2][0,10]
            "ldr q14, [x11, 208]\r\n"                                   // B[15,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[15,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[15,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[15,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,16] [0,0]
            "add x11, x1, #944\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[16,2][0,7]
            "add x11, x1, #1224\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[16,2][0,9]
            "ldr q12, [x11, 224]\r\n"                                   // B[16,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[16,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[16,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[16,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,17] [0,0]
            "add x11, x1, #768\r\n"                                     // 
            "ldr q2, [x11, 0]\r\n"                                      // B[17,2][0,1]
            "add x11, x1, #1088\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[17,2][0,8]
            "add x11, x1, #1352\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[17,2][0,10]
            "ldr q14, [x11, 208]\r\n"                                   // B[17,2][0,13]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[17,2][0,1]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[17,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[17,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[17,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,18] [0,0]
            "add x11, x1, #816\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[18,2][0,2]
            "ldr q8, [x11, 136]\r\n"                                    // B[18,2][0,7]
            "add x11, x1, #1232\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[18,2][0,9]
            "ldr q12, [x11, 224]\r\n"                                   // B[18,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[18,2][0,2]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[18,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[18,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[18,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,19] [0,0]
            "add x11, x1, #824\r\n"                                     // 
            "ldr q3, [x11, 0]\r\n"                                      // B[19,2][0,2]
            "ldr q5, [x11, 32]\r\n"                                     // B[19,2][0,4]
            "ldr q8, [x11, 136]\r\n"                                    // B[19,2][0,7]
            "add x11, x1, #1240\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[19,2][0,9]
            "ldr q12, [x11, 224]\r\n"                                   // B[19,2][0,11]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[19,2][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[19,2][0,4]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[19,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[19,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[19,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=20)
              "add x11, x0, #1280\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,20] [0,0]
            "add x11, x1, #968\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[20,2][0,7]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[20,2][0,7]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=21)
              "add x11, x0, #1344\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,21] [0,0]
            "add x11, x1, #1096\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[21,2][0,8]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[21,2][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=22)
              "add x11, x0, #1408\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,22] [0,0]
            "add x11, x1, #976\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[22,2][0,7]
            "add x11, x1, #1248\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[22,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[22,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[22,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=23)
              "add x11, x0, #1472\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,23] [0,0]
            "add x11, x1, #1104\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[23,2][0,8]
            "add x11, x1, #1360\r\n"                                    // 
            "ldr q11, [x11, 0]\r\n"                                     // B[23,2][0,10]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[23,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[23,2][0,10]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=24)
              "add x11, x0, #1536\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,24] [0,0]
            "add x11, x1, #984\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[24,2][0,7]
            "add x11, x1, #1256\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[24,2][0,9]
            "ldr q12, [x11, 216]\r\n"                                   // B[24,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[24,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[24,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[24,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=25)
              "add x11, x0, #1600\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,25] [0,0]
            "add x11, x1, #1112\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[25,2][0,8]
            "add x11, x1, #1568\r\n"                                    // 
            "ldr q14, [x11, 0]\r\n"                                     // B[25,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[25,2][0,8]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[25,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "add x11, x0, #1664\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,26] [0,0]
            "add x11, x1, #992\r\n"                                     // 
            "ldr q8, [x11, 0]\r\n"                                      // B[26,2][0,7]
            "add x11, x1, #1264\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[26,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[26,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[26,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "add x11, x0, #1728\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,27] [0,0]
            "add x11, x1, #1120\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[27,2][0,8]
            "ldr q11, [x11, 248]\r\n"                                   // B[27,2][0,10]
            "add x11, x1, #1576\r\n"                                    // 
            "ldr q14, [x11, 0]\r\n"                                     // B[27,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[27,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[27,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[27,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,28] [0,0]
            "add x11, x1, #1000\r\n"                                    // 
            "ldr q8, [x11, 0]\r\n"                                      // B[28,2][0,7]
            "add x11, x1, #1272\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[28,2][0,9]
            "ldr q12, [x11, 208]\r\n"                                   // B[28,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[28,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[28,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[28,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,29] [0,0]
            "add x11, x1, #1008\r\n"                                    // 
            "ldr q8, [x11, 0]\r\n"                                      // B[29,2][0,7]
            "add x11, x1, #1280\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[29,2][0,9]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[29,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[29,2][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "add x11, x0, #1920\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,30] [0,0]
            "add x11, x1, #1128\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[30,2][0,8]
            "ldr q11, [x11, 248]\r\n"                                   // B[30,2][0,10]
            "add x11, x1, #1584\r\n"                                    // 
            "ldr q14, [x11, 0]\r\n"                                     // B[30,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[30,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[30,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[30,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,31] [0,0]
            "add x11, x1, #1016\r\n"                                    // 
            "ldr q8, [x11, 0]\r\n"                                      // B[31,2][0,7]
            "add x11, x1, #1288\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[31,2][0,9]
            "ldr q12, [x11, 200]\r\n"                                   // B[31,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[31,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[31,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[31,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,32] [0,0]
            "add x11, x1, #1136\r\n"                                    // 
            "ldr q9, [x11, 0]\r\n"                                      // B[32,2][0,8]
            "ldr q11, [x11, 248]\r\n"                                   // B[32,2][0,10]
            "add x11, x1, #1592\r\n"                                    // 
            "ldr q14, [x11, 0]\r\n"                                     // B[32,2][0,13]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[32,2][0,8]
            "fmla v28.2d, v0.2d, v11.1d[0]\r\n"                         // C[0:2,10] += A[0:2,0]*B[32,2][0,10]
            "fmla v31.2d, v0.2d, v14.1d[0]\r\n"                         // C[0:2,13] += A[0:2,0]*B[32,2][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,33] [0,0]
            "add x11, x1, #1024\r\n"                                    // 
            "ldr q8, [x11, 0]\r\n"                                      // B[33,2][0,7]
            "add x11, x1, #1296\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[33,2][0,9]
            "ldr q12, [x11, 200]\r\n"                                   // B[33,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[33,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[33,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[33,2][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,34] [0,0]
            "add x11, x1, #1032\r\n"                                    // 
            "ldr q8, [x11, 0]\r\n"                                      // B[34,2][0,7]
            "add x11, x1, #1304\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[34,2][0,9]
            "ldr q12, [x11, 200]\r\n"                                   // B[34,2][0,11]
            "fmla v25.2d, v0.2d, v8.1d[0]\r\n"                          // C[0:2,7] += A[0:2,0]*B[34,2][0,7]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[34,2][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[34,2][0,11]
            // Store C register block @ (d=0,r=0)
            "str q18, [x2, 0]\r\n"                                      // C [0,0] [0,0]
            "str q19, [x2, 64]\r\n"                                     // C [0,0] [0,1]
            "str q20, [x2, 128]\r\n"                                    // C [0,0] [0,2]
            "str q21, [x2, 192]\r\n"                                    // C [0,0] [0,3]
            "add x11, x2, #256\r\n"                                     // 
            "str q22, [x11, 0]\r\n"                                     // C [0,0] [0,4]
            "str q23, [x11, 64]\r\n"                                    // C [0,0] [0,5]
            "str q24, [x11, 128]\r\n"                                   // C [0,0] [0,6]
            "str q25, [x11, 192]\r\n"                                   // C [0,0] [0,7]
            "add x11, x2, #512\r\n"                                     // 
            "str q26, [x11, 0]\r\n"                                     // C [0,0] [0,8]
            "str q27, [x11, 64]\r\n"                                    // C [0,0] [0,9]
            "str q28, [x11, 128]\r\n"                                   // C [0,0] [0,10]
            "str q29, [x11, 192]\r\n"                                   // C [0,0] [0,11]
            "add x11, x2, #768\r\n"                                     // 
            "str q30, [x11, 0]\r\n"                                     // C [0,0] [0,12]
            "str q31, [x11, 64]\r\n"                                    // C [0,0] [0,13]
          "add x2, x2, #896\r\n"                                      // Move C to (d=0,r=1)
            // zero registers
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
              "ldr q0, [x0, 0]\r\n"                                       // A [0,0] [0,0]
            "add x11, x1, #1600\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[0,3][0,0]
            "ldr q3, [x11, 224]\r\n"                                    // B[0,3][0,2]
            "add x11, x1, #1936\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[0,3][0,4]
            "ldr q7, [x11, 176]\r\n"                                    // B[0,3][0,6]
            "add x11, x1, #2240\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[0,3][0,9]
            "ldr q12, [x11, 72]\r\n"                                    // B[0,3][0,11]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[0,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[0,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[0,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[0,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[0,3][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[0,3][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "ldr q0, [x0, 64]\r\n"                                      // A [0,1] [0,0]
            "add x11, x1, #1752\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[1,3][0,1]
            "add x11, x1, #2056\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[1,3][0,5]
            "ldr q9, [x11, 152]\r\n"                                    // B[1,3][0,8]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[1,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[1,3][0,5]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[1,3][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "ldr q0, [x0, 128]\r\n"                                     // A [0,2] [0,0]
            "add x11, x1, #1608\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[2,3][0,0]
            "ldr q3, [x11, 224]\r\n"                                    // B[2,3][0,2]
            "add x11, x1, #1944\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[2,3][0,4]
            "ldr q7, [x11, 176]\r\n"                                    // B[2,3][0,6]
            "add x11, x1, #2248\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[2,3][0,9]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[2,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[2,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[2,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[2,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[2,3][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "ldr q0, [x0, 192]\r\n"                                     // A [0,3] [0,0]
            "add x11, x1, #1616\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[3,3][0,0]
            "ldr q3, [x11, 224]\r\n"                                    // B[3,3][0,2]
            "add x11, x1, #1952\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[3,3][0,4]
            "ldr q7, [x11, 176]\r\n"                                    // B[3,3][0,6]
            "add x11, x1, #2256\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[3,3][0,9]
            "ldr q12, [x11, 64]\r\n"                                    // B[3,3][0,11]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[3,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[3,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[3,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[3,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[3,3][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[3,3][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "add x11, x0, #256\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,4] [0,0]
            "add x11, x1, #1624\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[4,3][0,0]
            "add x11, x1, #1960\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[4,3][0,4]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[4,3][0,0]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[4,3][0,4]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "add x11, x0, #320\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,5] [0,0]
            "add x11, x1, #1760\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[5,3][0,1]
            "add x11, x1, #2064\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[5,3][0,5]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[5,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[5,3][0,5]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "add x11, x0, #384\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,6] [0,0]
            "add x11, x1, #1632\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[6,3][0,0]
            "ldr q3, [x11, 216]\r\n"                                    // B[6,3][0,2]
            "add x11, x1, #1968\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[6,3][0,4]
            "ldr q7, [x11, 168]\r\n"                                    // B[6,3][0,6]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[6,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[6,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[6,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[6,3][0,6]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "add x11, x0, #448\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,7] [0,0]
            "add x11, x1, #1768\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[7,3][0,1]
            "add x11, x1, #2072\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[7,3][0,5]
            "ldr q9, [x11, 144]\r\n"                                    // B[7,3][0,8]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[7,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[7,3][0,5]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[7,3][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "add x11, x0, #512\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,8] [0,0]
            "add x11, x1, #1640\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[8,3][0,0]
            "ldr q3, [x11, 216]\r\n"                                    // B[8,3][0,2]
            "add x11, x1, #1976\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[8,3][0,4]
            "ldr q7, [x11, 168]\r\n"                                    // B[8,3][0,6]
            "add x11, x1, #2264\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[8,3][0,9]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[8,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[8,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[8,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[8,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[8,3][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "add x11, x0, #576\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,9] [0,0]
            "add x11, x1, #1648\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[9,3][0,0]
            "ldr q3, [x11, 216]\r\n"                                    // B[9,3][0,2]
            "add x11, x1, #1984\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[9,3][0,4]
            "ldr q7, [x11, 168]\r\n"                                    // B[9,3][0,6]
            "add x11, x1, #2272\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[9,3][0,9]
            "ldr q12, [x11, 56]\r\n"                                    // B[9,3][0,11]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[9,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[9,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[9,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[9,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[9,3][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[9,3][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "add x11, x0, #704\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,11] [0,0]
            "add x11, x1, #1656\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[11,3][0,0]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[11,3][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "add x11, x0, #768\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,12] [0,0]
            "add x11, x1, #1776\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[12,3][0,1]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[12,3][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "add x11, x0, #832\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,13] [0,0]
            "add x11, x1, #1664\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[13,3][0,0]
            "ldr q3, [x11, 208]\r\n"                                    // B[13,3][0,2]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[13,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[13,3][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "add x11, x0, #896\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,14] [0,0]
            "add x11, x1, #1672\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[14,3][0,0]
            "add x11, x1, #1992\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[14,3][0,4]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[14,3][0,0]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[14,3][0,4]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "add x11, x0, #960\r\n"                                     // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,15] [0,0]
            "add x11, x1, #1784\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[15,3][0,1]
            "add x11, x1, #2080\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[15,3][0,5]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[15,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[15,3][0,5]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "add x11, x0, #1024\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,16] [0,0]
            "add x11, x1, #1680\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[16,3][0,0]
            "ldr q3, [x11, 200]\r\n"                                    // B[16,3][0,2]
            "add x11, x1, #2000\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[16,3][0,4]
            "ldr q7, [x11, 160]\r\n"                                    // B[16,3][0,6]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[16,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[16,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[16,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[16,3][0,6]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "add x11, x0, #1088\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,17] [0,0]
            "add x11, x1, #1792\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[17,3][0,1]
            "add x11, x1, #2088\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[17,3][0,5]
            "ldr q9, [x11, 136]\r\n"                                    // B[17,3][0,8]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[17,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[17,3][0,5]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[17,3][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "add x11, x0, #1152\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,18] [0,0]
            "add x11, x1, #1688\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[18,3][0,0]
            "ldr q3, [x11, 200]\r\n"                                    // B[18,3][0,2]
            "add x11, x1, #2008\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[18,3][0,4]
            "ldr q7, [x11, 160]\r\n"                                    // B[18,3][0,6]
            "add x11, x1, #2280\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[18,3][0,9]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[18,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[18,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[18,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[18,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[18,3][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "add x11, x0, #1216\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,19] [0,0]
            "add x11, x1, #1696\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[19,3][0,0]
            "ldr q3, [x11, 200]\r\n"                                    // B[19,3][0,2]
            "add x11, x1, #2016\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[19,3][0,4]
            "ldr q7, [x11, 160]\r\n"                                    // B[19,3][0,6]
            "add x11, x1, #2288\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[19,3][0,9]
            "ldr q12, [x11, 48]\r\n"                                    // B[19,3][0,11]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[19,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[19,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[19,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[19,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[19,3][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[19,3][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "add x11, x0, #1664\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,26] [0,0]
            "add x11, x1, #1704\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[26,3][0,0]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[26,3][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "add x11, x0, #1728\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,27] [0,0]
            "add x11, x1, #1800\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[27,3][0,1]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[27,3][0,1]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "add x11, x0, #1792\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,28] [0,0]
            "add x11, x1, #1712\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[28,3][0,0]
            "ldr q3, [x11, 192]\r\n"                                    // B[28,3][0,2]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[28,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[28,3][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "add x11, x0, #1856\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,29] [0,0]
            "add x11, x1, #1720\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[29,3][0,0]
            "add x11, x1, #2024\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[29,3][0,4]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[29,3][0,0]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[29,3][0,4]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "add x11, x0, #1920\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,30] [0,0]
            "add x11, x1, #1808\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[30,3][0,1]
            "add x11, x1, #2096\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[30,3][0,5]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[30,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[30,3][0,5]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "add x11, x0, #1984\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,31] [0,0]
            "add x11, x1, #1728\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[31,3][0,0]
            "ldr q3, [x11, 184]\r\n"                                    // B[31,3][0,2]
            "add x11, x1, #2032\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[31,3][0,4]
            "ldr q7, [x11, 152]\r\n"                                    // B[31,3][0,6]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[31,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[31,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[31,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[31,3][0,6]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "add x11, x0, #2048\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,32] [0,0]
            "add x11, x1, #1816\r\n"                                    // 
            "ldr q2, [x11, 0]\r\n"                                      // B[32,3][0,1]
            "add x11, x1, #2104\r\n"                                    // 
            "ldr q6, [x11, 0]\r\n"                                      // B[32,3][0,5]
            "ldr q9, [x11, 128]\r\n"                                    // B[32,3][0,8]
            "fmla v19.2d, v0.2d, v2.1d[0]\r\n"                          // C[0:2,1] += A[0:2,0]*B[32,3][0,1]
            "fmla v23.2d, v0.2d, v6.1d[0]\r\n"                          // C[0:2,5] += A[0:2,0]*B[32,3][0,5]
            "fmla v26.2d, v0.2d, v9.1d[0]\r\n"                          // C[0:2,8] += A[0:2,0]*B[32,3][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "add x11, x0, #2112\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,33] [0,0]
            "add x11, x1, #1736\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[33,3][0,0]
            "ldr q3, [x11, 184]\r\n"                                    // B[33,3][0,2]
            "add x11, x1, #2040\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[33,3][0,4]
            "ldr q7, [x11, 152]\r\n"                                    // B[33,3][0,6]
            "add x11, x1, #2296\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[33,3][0,9]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[33,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[33,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[33,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[33,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[33,3][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "add x11, x0, #2176\r\n"                                    // 
              "ldr q0, [x11, 0]\r\n"                                      // A [0,34] [0,0]
            "add x11, x1, #1744\r\n"                                    // 
            "ldr q1, [x11, 0]\r\n"                                      // B[34,3][0,0]
            "ldr q3, [x11, 184]\r\n"                                    // B[34,3][0,2]
            "add x11, x1, #2048\r\n"                                    // 
            "ldr q5, [x11, 0]\r\n"                                      // B[34,3][0,4]
            "ldr q7, [x11, 152]\r\n"                                    // B[34,3][0,6]
            "add x11, x1, #2304\r\n"                                    // 
            "ldr q10, [x11, 0]\r\n"                                     // B[34,3][0,9]
            "ldr q12, [x11, 40]\r\n"                                    // B[34,3][0,11]
            "fmla v18.2d, v0.2d, v1.1d[0]\r\n"                          // C[0:2,0] += A[0:2,0]*B[34,3][0,0]
            "fmla v20.2d, v0.2d, v3.1d[0]\r\n"                          // C[0:2,2] += A[0:2,0]*B[34,3][0,2]
            "fmla v22.2d, v0.2d, v5.1d[0]\r\n"                          // C[0:2,4] += A[0:2,0]*B[34,3][0,4]
            "fmla v24.2d, v0.2d, v7.1d[0]\r\n"                          // C[0:2,6] += A[0:2,0]*B[34,3][0,6]
            "fmla v27.2d, v0.2d, v10.1d[0]\r\n"                         // C[0:2,9] += A[0:2,0]*B[34,3][0,9]
            "fmla v29.2d, v0.2d, v12.1d[0]\r\n"                         // C[0:2,11] += A[0:2,0]*B[34,3][0,11]
            // Store C register block @ (d=0,r=0)
            "str q18, [x2, 0]\r\n"                                      // C [0,0] [0,0]
            "str q19, [x2, 64]\r\n"                                     // C [0,0] [0,1]
            "str q20, [x2, 128]\r\n"                                    // C [0,0] [0,2]
            "str q21, [x2, 192]\r\n"                                    // C [0,0] [0,3]
            "add x11, x2, #256\r\n"                                     // 
            "str q22, [x11, 0]\r\n"                                     // C [0,0] [0,4]
            "str q23, [x11, 64]\r\n"                                    // C [0,0] [0,5]
            "str q24, [x11, 128]\r\n"                                   // C [0,0] [0,6]
            "str q25, [x11, 192]\r\n"                                   // C [0,0] [0,7]
            "add x11, x2, #512\r\n"                                     // 
            "str q26, [x11, 0]\r\n"                                     // C [0,0] [0,8]
            "str q27, [x11, 64]\r\n"                                    // C [0,0] [0,9]
            "str q28, [x11, 128]\r\n"                                   // C [0,0] [0,10]
            "str q29, [x11, 192]\r\n"                                   // C [0,0] [0,11]
            "add x11, x2, #768\r\n"                                     // 
            "str q30, [x11, 0]\r\n"                                     // C [0,0] [0,12]
            "str q31, [x11, 64]\r\n"                                    // C [0,0] [0,13]
        "add x0, x0, #16\r\n"                                       // Move A to (d=1,r=0)
        "add x2, x2, #-2672\r\n"                                    // Move C to (d=1,r=-3)
        "add x12, x12, #1\r\n"
        "cmp x12, #4\r\n"
        "b.lo LOOP_TOP_0_%=\r\n"

    : : "m"(A), "m"(B), "m"(C) : "r0","r11","r12","r2","v0","v1","v10","v11","v12","v13","v14","v18","v19","v2","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v3","v30","v31","v4","v5","v6","v7","v8","v9");

};