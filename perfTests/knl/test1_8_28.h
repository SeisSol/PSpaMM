
void test1_8_28 (const double* A, const double* B, double* C) {
  __asm__ __volatile__(
    "movq %0, %%rdi\n\t"
    "movq %1, %%rsi\n\t"
    "movq %2, %%rdx\n\t"
      // unrolled_8x56x56
        // for %%r12 <- 0:1:1)
        "movq $0, %%r12\r\n"
        "LOOP_TOP_0_%=:\r\n"
          // Unrolling over bn and bk
            // zero registers
            "vpxord %%zmm4, %%zmm4, %%zmm4\r\n"
            "vpxord %%zmm5, %%zmm5, %%zmm5\r\n"
            "vpxord %%zmm6, %%zmm6, %%zmm6\r\n"
            "vpxord %%zmm7, %%zmm7, %%zmm7\r\n"
            "vpxord %%zmm8, %%zmm8, %%zmm8\r\n"
            "vpxord %%zmm9, %%zmm9, %%zmm9\r\n"
            "vpxord %%zmm10, %%zmm10, %%zmm10\r\n"
            "vpxord %%zmm11, %%zmm11, %%zmm11\r\n"
            "vpxord %%zmm12, %%zmm12, %%zmm12\r\n"
            "vpxord %%zmm13, %%zmm13, %%zmm13\r\n"
            "vpxord %%zmm14, %%zmm14, %%zmm14\r\n"
            "vpxord %%zmm15, %%zmm15, %%zmm15\r\n"
            "vpxord %%zmm16, %%zmm16, %%zmm16\r\n"
            "vpxord %%zmm17, %%zmm17, %%zmm17\r\n"
            "vpxord %%zmm18, %%zmm18, %%zmm18\r\n"
            "vpxord %%zmm19, %%zmm19, %%zmm19\r\n"
            "vpxord %%zmm20, %%zmm20, %%zmm20\r\n"
            "vpxord %%zmm21, %%zmm21, %%zmm21\r\n"
            "vpxord %%zmm22, %%zmm22, %%zmm22\r\n"
            "vpxord %%zmm23, %%zmm23, %%zmm23\r\n"
            "vpxord %%zmm24, %%zmm24, %%zmm24\r\n"
            "vpxord %%zmm25, %%zmm25, %%zmm25\r\n"
            "vpxord %%zmm26, %%zmm26, %%zmm26\r\n"
            "vpxord %%zmm27, %%zmm27, %%zmm27\r\n"
            "vpxord %%zmm28, %%zmm28, %%zmm28\r\n"
            "vpxord %%zmm29, %%zmm29, %%zmm29\r\n"
            "vpxord %%zmm30, %%zmm30, %%zmm30\r\n"
            "vpxord %%zmm31, %%zmm31, %%zmm31\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "vmovapd 64(%%rdi), %%zmm0\r\n"                             // A [0,1] [0,0]
            "vfmadd231pd 496(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[1,0][0,12]
            "vfmadd231pd 920(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"       // C[0:8,20] += A[0:8,0]*B[1,0][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "vmovapd 128(%%rdi), %%zmm0\r\n"                            // A [0,2] [0,0]
            "vfmadd231pd 288(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"       // C[0:8,7] += A[0:8,0]*B[2,0][0,7]
            "vfmadd231pd 584(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"       // C[0:8,14] += A[0:8,0]*B[2,0][0,14]
            "vfmadd231pd 624(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"       // C[0:8,15] += A[0:8,0]*B[2,0][0,15]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=3)
              "vmovapd 192(%%rdi), %%zmm0\r\n"                            // A [0,3] [0,0]
            "vfmadd231pd 816(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[3,0][0,19]
            "vfmadd231pd 1016(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[3,0][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "vmovapd 256(%%rdi), %%zmm0\r\n"                            // A [0,4] [0,0]
            "vfmadd231pd 352(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[4,0][0,9]
            "vfmadd231pd 696(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[4,0][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "vmovapd 320(%%rdi), %%zmm0\r\n"                            // A [0,5] [0,0]
            "vfmadd231pd 208(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"        // C[0:8,5] += A[0:8,0]*B[5,0][0,5]
            "vfmadd231pd 824(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[5,0][0,19]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "vmovapd 384(%%rdi), %%zmm0\r\n"                            // A [0,6] [0,0]
            "vfmadd231pd 704(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[6,0][0,17]
            "vfmadd231pd 760(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[6,0][0,18]
            "vfmadd231pd 832(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[6,0][0,19]
            "vfmadd231pd 984(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"       // C[0:8,22] += A[0:8,0]*B[6,0][0,22]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "vmovapd 448(%%rdi), %%zmm0\r\n"                            // A [0,7] [0,0]
            "vfmadd231pd 136(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[7,0][0,3]
            "vfmadd231pd 248(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"       // C[0:8,6] += A[0:8,0]*B[7,0][0,6]
            "vfmadd231pd 840(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[7,0][0,19]
            "vfmadd231pd 952(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"       // C[0:8,21] += A[0:8,0]*B[7,0][0,21]
            "vfmadd231pd 1096(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[7,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "vmovapd 512(%%rdi), %%zmm0\r\n"                            // A [0,8] [0,0]
            "vfmadd231pd 416(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"       // C[0:8,10] += A[0:8,0]*B[8,0][0,10]
            "vfmadd231pd 848(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[8,0][0,19]
            "vfmadd231pd 1104(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[8,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "vmovapd 576(%%rdi), %%zmm0\r\n"                            // A [0,9] [0,0]
            "vfmadd231pd 360(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[9,0][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "vmovapd 640(%%rdi), %%zmm0\r\n"                            // A [0,10] [0,0]
            "vfmadd231pd 96(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"         // C[0:8,2] += A[0:8,0]*B[10,0][0,2]
            "vfmadd231pd 184(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"        // C[0:8,4] += A[0:8,0]*B[10,0][0,4]
            "vfmadd231pd 960(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"       // C[0:8,21] += A[0:8,0]*B[10,0][0,21]
            "vfmadd231pd 1176(%%rsi)%{1to8%}, %%zmm0, %%zmm31\r\n"      // C[0:8,27] += A[0:8,0]*B[10,0][0,27]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "vmovapd 704(%%rdi), %%zmm0\r\n"                            // A [0,11] [0,0]
            "vfmadd231pd 504(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[11,0][0,12]
            "vfmadd231pd 992(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"       // C[0:8,22] += A[0:8,0]*B[11,0][0,22]
            "vfmadd231pd 1024(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[11,0][0,23]
            "vfmadd231pd 1144(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[11,0][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "vmovapd 768(%%rdi), %%zmm0\r\n"                            // A [0,12] [0,0]
            "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"          // C[0:8,0] += A[0:8,0]*B[12,0][0,0]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "vmovapd 832(%%rdi), %%zmm0\r\n"                            // A [0,13] [0,0]
            "vfmadd231pd 368(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[13,0][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "vmovapd 896(%%rdi), %%zmm0\r\n"                            // A [0,14] [0,0]
            "vfmadd231pd 552(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"       // C[0:8,13] += A[0:8,0]*B[14,0][0,13]
            "vfmadd231pd 632(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"       // C[0:8,15] += A[0:8,0]*B[14,0][0,15]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=15)
              "vmovapd 960(%%rdi), %%zmm0\r\n"                            // A [0,15] [0,0]
            "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"          // C[0:8,0] += A[0:8,0]*B[15,0][0,0]
            "vfmadd231pd 376(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[15,0][0,9]
            "vfmadd231pd 512(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[15,0][0,12]
            "vfmadd231pd 768(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[15,0][0,18]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "vmovapd 1024(%%rdi), %%zmm0\r\n"                           // A [0,16] [0,0]
            "vfmadd231pd 328(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"       // C[0:8,8] += A[0:8,0]*B[16,0][0,8]
            "vfmadd231pd 712(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[16,0][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "vmovapd 1088(%%rdi), %%zmm0\r\n"                           // A [0,17] [0,0]
            "vfmadd231pd 144(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[17,0][0,3]
            "vfmadd231pd 216(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"        // C[0:8,5] += A[0:8,0]*B[17,0][0,5]
            "vfmadd231pd 592(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"       // C[0:8,14] += A[0:8,0]*B[17,0][0,14]
            "vfmadd231pd 664(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"       // C[0:8,16] += A[0:8,0]*B[17,0][0,16]
            "vfmadd231pd 720(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[17,0][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=18)
              "vmovapd 1152(%%rdi), %%zmm0\r\n"                           // A [0,18] [0,0]
            "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"         // C[0:8,1] += A[0:8,0]*B[18,0][0,1]
            "vfmadd231pd 520(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[18,0][0,12]
            "vfmadd231pd 1000(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"      // C[0:8,22] += A[0:8,0]*B[18,0][0,22]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "vmovapd 1216(%%rdi), %%zmm0\r\n"                           // A [0,19] [0,0]
            "vfmadd231pd 152(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[19,0][0,3]
            "vfmadd231pd 456(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"       // C[0:8,11] += A[0:8,0]*B[19,0][0,11]
            "vfmadd231pd 728(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[19,0][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=20)
              "vmovapd 1280(%%rdi), %%zmm0\r\n"                           // A [0,20] [0,0]
            "vfmadd231pd 336(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"       // C[0:8,8] += A[0:8,0]*B[20,0][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=22)
              "vmovapd 1408(%%rdi), %%zmm0\r\n"                           // A [0,22] [0,0]
            "vfmadd231pd 192(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"        // C[0:8,4] += A[0:8,0]*B[22,0][0,4]
            "vfmadd231pd 256(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"       // C[0:8,6] += A[0:8,0]*B[22,0][0,6]
            "vfmadd231pd 1112(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[22,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=23)
              "vmovapd 1472(%%rdi), %%zmm0\r\n"                           // A [0,23] [0,0]
            "vfmadd231pd 640(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"       // C[0:8,15] += A[0:8,0]*B[23,0][0,15]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=24)
              "vmovapd 1536(%%rdi), %%zmm0\r\n"                           // A [0,24] [0,0]
            "vfmadd231pd 528(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[24,0][0,12]
            "vfmadd231pd 1120(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[24,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=25)
              "vmovapd 1600(%%rdi), %%zmm0\r\n"                           // A [0,25] [0,0]
            "vfmadd231pd 64(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"         // C[0:8,1] += A[0:8,0]*B[25,0][0,1]
            "vfmadd231pd 464(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"       // C[0:8,11] += A[0:8,0]*B[25,0][0,11]
            "vfmadd231pd 856(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[25,0][0,19]
            "vfmadd231pd 1056(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[25,0][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "vmovapd 1664(%%rdi), %%zmm0\r\n"                           // A [0,26] [0,0]
            "vfmadd231pd 384(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[26,0][0,9]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "vmovapd 1728(%%rdi), %%zmm0\r\n"                           // A [0,27] [0,0]
            "vfmadd231pd 296(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"       // C[0:8,7] += A[0:8,0]*B[27,0][0,7]
            "vfmadd231pd 536(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[27,0][0,12]
            "vfmadd231pd 864(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[27,0][0,19]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "vmovapd 1792(%%rdi), %%zmm0\r\n"                           // A [0,28] [0,0]
            "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"         // C[0:8,0] += A[0:8,0]*B[28,0][0,0]
            "vfmadd231pd 304(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"       // C[0:8,7] += A[0:8,0]*B[28,0][0,7]
            "vfmadd231pd 872(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[28,0][0,19]
            "vfmadd231pd 1032(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[28,0][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "vmovapd 1856(%%rdi), %%zmm0\r\n"                           // A [0,29] [0,0]
            "vfmadd231pd 880(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[29,0][0,19]
            "vfmadd231pd 1064(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[29,0][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "vmovapd 1920(%%rdi), %%zmm0\r\n"                           // A [0,30] [0,0]
            "vfmadd231pd 72(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"         // C[0:8,1] += A[0:8,0]*B[30,0][0,1]
            "vfmadd231pd 472(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"       // C[0:8,11] += A[0:8,0]*B[30,0][0,11]
            "vfmadd231pd 1008(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"      // C[0:8,22] += A[0:8,0]*B[30,0][0,22]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "vmovapd 1984(%%rdi), %%zmm0\r\n"                           // A [0,31] [0,0]
            "vfmadd231pd 264(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"       // C[0:8,6] += A[0:8,0]*B[31,0][0,6]
            "vfmadd231pd 672(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"       // C[0:8,16] += A[0:8,0]*B[31,0][0,16]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=32)
              "vmovapd 2048(%%rdi), %%zmm0\r\n"                           // A [0,32] [0,0]
            "vfmadd231pd 160(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[32,0][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "vmovapd 2112(%%rdi), %%zmm0\r\n"                           // A [0,33] [0,0]
            "vfmadd231pd 272(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"       // C[0:8,6] += A[0:8,0]*B[33,0][0,6]
            "vfmadd231pd 392(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[33,0][0,9]
            "vfmadd231pd 560(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"       // C[0:8,13] += A[0:8,0]*B[33,0][0,13]
            "vfmadd231pd 736(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[33,0][0,17]
            "vfmadd231pd 888(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[33,0][0,19]
            "vfmadd231pd 928(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"       // C[0:8,20] += A[0:8,0]*B[33,0][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "vmovapd 2176(%%rdi), %%zmm0\r\n"                           // A [0,34] [0,0]
            "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"         // C[0:8,0] += A[0:8,0]*B[34,0][0,0]
            "vfmadd231pd 168(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[34,0][0,3]
            "vfmadd231pd 312(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"       // C[0:8,7] += A[0:8,0]*B[34,0][0,7]
            "vfmadd231pd 648(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"       // C[0:8,15] += A[0:8,0]*B[34,0][0,15]
            "vfmadd231pd 1128(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[34,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=35)
              "vmovapd 2240(%%rdi), %%zmm0\r\n"                           // A [0,35] [0,0]
            "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"         // C[0:8,0] += A[0:8,0]*B[35,0][0,0]
            "vfmadd231pd 544(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"       // C[0:8,12] += A[0:8,0]*B[35,0][0,12]
            "vfmadd231pd 744(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[35,0][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=36)
              "vmovapd 2304(%%rdi), %%zmm0\r\n"                           // A [0,36] [0,0]
            "vfmadd231pd 344(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"       // C[0:8,8] += A[0:8,0]*B[36,0][0,8]
            "vfmadd231pd 400(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[36,0][0,9]
            "vfmadd231pd 1072(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[36,0][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=37)
              "vmovapd 2368(%%rdi), %%zmm0\r\n"                           // A [0,37] [0,0]
            "vfmadd231pd 224(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"        // C[0:8,5] += A[0:8,0]*B[37,0][0,5]
            "vfmadd231pd 408(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"       // C[0:8,9] += A[0:8,0]*B[37,0][0,9]
            "vfmadd231pd 424(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"       // C[0:8,10] += A[0:8,0]*B[37,0][0,10]
            "vfmadd231pd 680(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"       // C[0:8,16] += A[0:8,0]*B[37,0][0,16]
            "vfmadd231pd 1152(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[37,0][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=38)
              "vmovapd 2432(%%rdi), %%zmm0\r\n"                           // A [0,38] [0,0]
            "vfmadd231pd 776(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[38,0][0,18]
            "vfmadd231pd 1160(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[38,0][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=39)
              "vmovapd 2496(%%rdi), %%zmm0\r\n"                           // A [0,39] [0,0]
            "vfmadd231pd 104(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"        // C[0:8,2] += A[0:8,0]*B[39,0][0,2]
            "vfmadd231pd 200(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"        // C[0:8,4] += A[0:8,0]*B[39,0][0,4]
            "vfmadd231pd 232(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"        // C[0:8,5] += A[0:8,0]*B[39,0][0,5]
            "vfmadd231pd 568(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"       // C[0:8,13] += A[0:8,0]*B[39,0][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=40)
              "vmovapd 2560(%%rdi), %%zmm0\r\n"                           // A [0,40] [0,0]
            "vfmadd231pd 80(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"         // C[0:8,1] += A[0:8,0]*B[40,0][0,1]
            "vfmadd231pd 600(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"       // C[0:8,14] += A[0:8,0]*B[40,0][0,14]
            "vfmadd231pd 784(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[40,0][0,18]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=41)
              "vmovapd 2624(%%rdi), %%zmm0\r\n"                           // A [0,41] [0,0]
            "vfmadd231pd 688(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"       // C[0:8,16] += A[0:8,0]*B[41,0][0,16]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=42)
              "vmovapd 2688(%%rdi), %%zmm0\r\n"                           // A [0,42] [0,0]
            "vfmadd231pd 176(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"        // C[0:8,3] += A[0:8,0]*B[42,0][0,3]
            "vfmadd231pd 320(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"       // C[0:8,7] += A[0:8,0]*B[42,0][0,7]
            "vfmadd231pd 480(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"       // C[0:8,11] += A[0:8,0]*B[42,0][0,11]
            "vfmadd231pd 752(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"       // C[0:8,17] += A[0:8,0]*B[42,0][0,17]
            "vfmadd231pd 1136(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[42,0][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=43)
              "vmovapd 2752(%%rdi), %%zmm0\r\n"                           // A [0,43] [0,0]
            "vfmadd231pd 896(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[43,0][0,19]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=44)
              "vmovapd 2816(%%rdi), %%zmm0\r\n"                           // A [0,44] [0,0]
            "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"         // C[0:8,0] += A[0:8,0]*B[44,0][0,0]
            "vfmadd231pd 88(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"         // C[0:8,1] += A[0:8,0]*B[44,0][0,1]
            "vfmadd231pd 488(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"       // C[0:8,11] += A[0:8,0]*B[44,0][0,11]
            "vfmadd231pd 968(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"       // C[0:8,21] += A[0:8,0]*B[44,0][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=45)
              "vmovapd 2880(%%rdi), %%zmm0\r\n"                           // A [0,45] [0,0]
            "vfmadd231pd 112(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"        // C[0:8,2] += A[0:8,0]*B[45,0][0,2]
            "vfmadd231pd 280(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"       // C[0:8,6] += A[0:8,0]*B[45,0][0,6]
            "vfmadd231pd 576(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"       // C[0:8,13] += A[0:8,0]*B[45,0][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=46)
              "vmovapd 2944(%%rdi), %%zmm0\r\n"                           // A [0,46] [0,0]
            "vfmadd231pd 432(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"       // C[0:8,10] += A[0:8,0]*B[46,0][0,10]
            "vfmadd231pd 904(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[46,0][0,19]
            "vfmadd231pd 936(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"       // C[0:8,20] += A[0:8,0]*B[46,0][0,20]
            "vfmadd231pd 1040(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[46,0][0,23]
            "vfmadd231pd 1080(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[46,0][0,24]
            "vfmadd231pd 1184(%%rsi)%{1to8%}, %%zmm0, %%zmm31\r\n"      // C[0:8,27] += A[0:8,0]*B[46,0][0,27]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=48)
              "vmovapd 3072(%%rdi), %%zmm0\r\n"                           // A [0,48] [0,0]
            "vfmadd231pd 240(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"        // C[0:8,5] += A[0:8,0]*B[48,0][0,5]
            "vfmadd231pd 792(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[48,0][0,18]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=49)
              "vmovapd 3136(%%rdi), %%zmm0\r\n"                           // A [0,49] [0,0]
            "vfmadd231pd 440(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"       // C[0:8,10] += A[0:8,0]*B[49,0][0,10]
            "vfmadd231pd 608(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"       // C[0:8,14] += A[0:8,0]*B[49,0][0,14]
            "vfmadd231pd 1048(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[49,0][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=50)
              "vmovapd 3200(%%rdi), %%zmm0\r\n"                           // A [0,50] [0,0]
            "vfmadd231pd 912(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"       // C[0:8,19] += A[0:8,0]*B[50,0][0,19]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=51)
              "vmovapd 3264(%%rdi), %%zmm0\r\n"                           // A [0,51] [0,0]
            "vfmadd231pd 656(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"       // C[0:8,15] += A[0:8,0]*B[51,0][0,15]
            "vfmadd231pd 800(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[51,0][0,18]
            "vfmadd231pd 1168(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[51,0][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=53)
              "vmovapd 3392(%%rdi), %%zmm0\r\n"                           // A [0,53] [0,0]
            "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"         // C[0:8,0] += A[0:8,0]*B[53,0][0,0]
            "vfmadd231pd 120(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"        // C[0:8,2] += A[0:8,0]*B[53,0][0,2]
            "vfmadd231pd 448(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"       // C[0:8,10] += A[0:8,0]*B[53,0][0,10]
            "vfmadd231pd 1088(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[53,0][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=54)
              "vmovapd 3456(%%rdi), %%zmm0\r\n"                           // A [0,54] [0,0]
            "vfmadd231pd 128(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"        // C[0:8,2] += A[0:8,0]*B[54,0][0,2]
            "vfmadd231pd 808(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"       // C[0:8,18] += A[0:8,0]*B[54,0][0,18]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=55)
              "vmovapd 3520(%%rdi), %%zmm0\r\n"                           // A [0,55] [0,0]
            "vfmadd231pd 616(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"       // C[0:8,14] += A[0:8,0]*B[55,0][0,14]
            "vfmadd231pd 944(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"       // C[0:8,20] += A[0:8,0]*B[55,0][0,20]
            "vfmadd231pd 976(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"       // C[0:8,21] += A[0:8,0]*B[55,0][0,21]
            // Store C register block @ (d=0,r=0)
            "vmovapd %%zmm4, 0(%%rdx)\r\n"                              // C [0,0] [0,0]
            "vmovapd %%zmm5, 64(%%rdx)\r\n"                             // C [0,0] [0,1]
            "vmovapd %%zmm6, 128(%%rdx)\r\n"                            // C [0,0] [0,2]
            "vmovapd %%zmm7, 192(%%rdx)\r\n"                            // C [0,0] [0,3]
            "vmovapd %%zmm8, 256(%%rdx)\r\n"                            // C [0,0] [0,4]
            "vmovapd %%zmm9, 320(%%rdx)\r\n"                            // C [0,0] [0,5]
            "vmovapd %%zmm10, 384(%%rdx)\r\n"                           // C [0,0] [0,6]
            "vmovapd %%zmm11, 448(%%rdx)\r\n"                           // C [0,0] [0,7]
            "vmovapd %%zmm12, 512(%%rdx)\r\n"                           // C [0,0] [0,8]
            "vmovapd %%zmm13, 576(%%rdx)\r\n"                           // C [0,0] [0,9]
            "vmovapd %%zmm14, 640(%%rdx)\r\n"                           // C [0,0] [0,10]
            "vmovapd %%zmm15, 704(%%rdx)\r\n"                           // C [0,0] [0,11]
            "vmovapd %%zmm16, 768(%%rdx)\r\n"                           // C [0,0] [0,12]
            "vmovapd %%zmm17, 832(%%rdx)\r\n"                           // C [0,0] [0,13]
            "vmovapd %%zmm18, 896(%%rdx)\r\n"                           // C [0,0] [0,14]
            "vmovapd %%zmm19, 960(%%rdx)\r\n"                           // C [0,0] [0,15]
            "vmovapd %%zmm20, 1024(%%rdx)\r\n"                          // C [0,0] [0,16]
            "vmovapd %%zmm21, 1088(%%rdx)\r\n"                          // C [0,0] [0,17]
            "vmovapd %%zmm22, 1152(%%rdx)\r\n"                          // C [0,0] [0,18]
            "vmovapd %%zmm23, 1216(%%rdx)\r\n"                          // C [0,0] [0,19]
            "vmovapd %%zmm24, 1280(%%rdx)\r\n"                          // C [0,0] [0,20]
            "vmovapd %%zmm25, 1344(%%rdx)\r\n"                          // C [0,0] [0,21]
            "vmovapd %%zmm26, 1408(%%rdx)\r\n"                          // C [0,0] [0,22]
            "vmovapd %%zmm27, 1472(%%rdx)\r\n"                          // C [0,0] [0,23]
            "vmovapd %%zmm28, 1536(%%rdx)\r\n"                          // C [0,0] [0,24]
            "vmovapd %%zmm29, 1600(%%rdx)\r\n"                          // C [0,0] [0,25]
            "vmovapd %%zmm30, 1664(%%rdx)\r\n"                          // C [0,0] [0,26]
            "vmovapd %%zmm31, 1728(%%rdx)\r\n"                          // C [0,0] [0,27]
          "addq $1792, %%rdx\r\n"                                     // Move C to (d=0,r=1)
            // zero registers
            "vpxord %%zmm4, %%zmm4, %%zmm4\r\n"
            "vpxord %%zmm5, %%zmm5, %%zmm5\r\n"
            "vpxord %%zmm6, %%zmm6, %%zmm6\r\n"
            "vpxord %%zmm7, %%zmm7, %%zmm7\r\n"
            "vpxord %%zmm8, %%zmm8, %%zmm8\r\n"
            "vpxord %%zmm9, %%zmm9, %%zmm9\r\n"
            "vpxord %%zmm10, %%zmm10, %%zmm10\r\n"
            "vpxord %%zmm11, %%zmm11, %%zmm11\r\n"
            "vpxord %%zmm12, %%zmm12, %%zmm12\r\n"
            "vpxord %%zmm13, %%zmm13, %%zmm13\r\n"
            "vpxord %%zmm14, %%zmm14, %%zmm14\r\n"
            "vpxord %%zmm15, %%zmm15, %%zmm15\r\n"
            "vpxord %%zmm16, %%zmm16, %%zmm16\r\n"
            "vpxord %%zmm17, %%zmm17, %%zmm17\r\n"
            "vpxord %%zmm18, %%zmm18, %%zmm18\r\n"
            "vpxord %%zmm19, %%zmm19, %%zmm19\r\n"
            "vpxord %%zmm20, %%zmm20, %%zmm20\r\n"
            "vpxord %%zmm21, %%zmm21, %%zmm21\r\n"
            "vpxord %%zmm22, %%zmm22, %%zmm22\r\n"
            "vpxord %%zmm23, %%zmm23, %%zmm23\r\n"
            "vpxord %%zmm24, %%zmm24, %%zmm24\r\n"
            "vpxord %%zmm25, %%zmm25, %%zmm25\r\n"
            "vpxord %%zmm26, %%zmm26, %%zmm26\r\n"
            "vpxord %%zmm27, %%zmm27, %%zmm27\r\n"
            "vpxord %%zmm28, %%zmm28, %%zmm28\r\n"
            "vpxord %%zmm29, %%zmm29, %%zmm29\r\n"
            "vpxord %%zmm30, %%zmm30, %%zmm30\r\n"
            "vpxord %%zmm31, %%zmm31, %%zmm31\r\n"
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=0)
              "vmovapd 0(%%rdi), %%zmm0\r\n"                              // A [0,0] [0,0]
            "vfmadd231pd 1224(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[0,1][0,1]
            "vfmadd231pd 1360(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"       // C[0:8,4] += A[0:8,0]*B[0,1][0,4]
            "vfmadd231pd 1528(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[0,1][0,8]
            "vfmadd231pd 1856(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"      // C[0:8,15] += A[0:8,0]*B[0,1][0,15]
            "vfmadd231pd 1880(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"      // C[0:8,16] += A[0:8,0]*B[0,1][0,16]
            "vfmadd231pd 2272(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[0,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=1)
              "vmovapd 64(%%rdi), %%zmm0\r\n"                             // A [0,1] [0,0]
            "vfmadd231pd 1696(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[1,1][0,12]
            "vfmadd231pd 2280(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[1,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=2)
              "vmovapd 128(%%rdi), %%zmm0\r\n"                            // A [0,2] [0,0]
            "vfmadd231pd 1288(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[2,1][0,2]
            "vfmadd231pd 1432(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[2,1][0,6]
            "vfmadd231pd 1800(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[2,1][0,14]
            "vfmadd231pd 2032(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"      // C[0:8,19] += A[0:8,0]*B[2,1][0,19]
            "vfmadd231pd 2288(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[2,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=4)
              "vmovapd 256(%%rdi), %%zmm0\r\n"                            // A [0,4] [0,0]
            "vfmadd231pd 1584(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"      // C[0:8,9] += A[0:8,0]*B[4,1][0,9]
            "vfmadd231pd 1704(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[4,1][0,12]
            "vfmadd231pd 1808(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[4,1][0,14]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=5)
              "vmovapd 320(%%rdi), %%zmm0\r\n"                            // A [0,5] [0,0]
            "vfmadd231pd 1296(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[5,1][0,2]
            "vfmadd231pd 1816(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[5,1][0,14]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=6)
              "vmovapd 384(%%rdi), %%zmm0\r\n"                            // A [0,6] [0,0]
            "vfmadd231pd 1440(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[6,1][0,6]
            "vfmadd231pd 1592(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"      // C[0:8,9] += A[0:8,0]*B[6,1][0,9]
            "vfmadd231pd 1824(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[6,1][0,14]
            "vfmadd231pd 1864(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"      // C[0:8,15] += A[0:8,0]*B[6,1][0,15]
            "vfmadd231pd 2176(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"      // C[0:8,22] += A[0:8,0]*B[6,1][0,22]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=7)
              "vmovapd 448(%%rdi), %%zmm0\r\n"                            // A [0,7] [0,0]
            "vfmadd231pd 1744(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[7,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=8)
              "vmovapd 512(%%rdi), %%zmm0\r\n"                            // A [0,8] [0,0]
            "vfmadd231pd 1752(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[8,1][0,13]
            "vfmadd231pd 1920(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[8,1][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=9)
              "vmovapd 576(%%rdi), %%zmm0\r\n"                            // A [0,9] [0,0]
            "vfmadd231pd 1304(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[9,1][0,2]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=10)
              "vmovapd 640(%%rdi), %%zmm0\r\n"                            // A [0,10] [0,0]
            "vfmadd231pd 2040(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"      // C[0:8,19] += A[0:8,0]*B[10,1][0,19]
            "vfmadd231pd 2104(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[10,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=11)
              "vmovapd 704(%%rdi), %%zmm0\r\n"                            // A [0,11] [0,0]
            "vfmadd231pd 1536(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[11,1][0,8]
            "vfmadd231pd 1968(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[11,1][0,18]
            "vfmadd231pd 2320(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[11,1][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=12)
              "vmovapd 768(%%rdi), %%zmm0\r\n"                            // A [0,12] [0,0]
            "vfmadd231pd 1368(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"       // C[0:8,4] += A[0:8,0]*B[12,1][0,4]
            "vfmadd231pd 1600(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"      // C[0:8,9] += A[0:8,0]*B[12,1][0,9]
            "vfmadd231pd 1888(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"      // C[0:8,16] += A[0:8,0]*B[12,1][0,16]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=13)
              "vmovapd 832(%%rdi), %%zmm0\r\n"                            // A [0,13] [0,0]
            "vfmadd231pd 1232(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[13,1][0,1]
            "vfmadd231pd 1928(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[13,1][0,17]
            "vfmadd231pd 2112(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[13,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=14)
              "vmovapd 896(%%rdi), %%zmm0\r\n"                            // A [0,14] [0,0]
            "vfmadd231pd 1616(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[14,1][0,10]
            "vfmadd231pd 2120(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[14,1][0,21]
            "vfmadd231pd 2200(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[14,1][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=16)
              "vmovapd 1024(%%rdi), %%zmm0\r\n"                           // A [0,16] [0,0]
            "vfmadd231pd 1240(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[16,1][0,1]
            "vfmadd231pd 1392(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"       // C[0:8,5] += A[0:8,0]*B[16,1][0,5]
            "vfmadd231pd 1896(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"      // C[0:8,16] += A[0:8,0]*B[16,1][0,16]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=17)
              "vmovapd 1088(%%rdi), %%zmm0\r\n"                           // A [0,17] [0,0]
            "vfmadd231pd 1544(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[17,1][0,8]
            "vfmadd231pd 1936(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[17,1][0,17]
            "vfmadd231pd 2296(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[17,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=19)
              "vmovapd 1216(%%rdi), %%zmm0\r\n"                           // A [0,19] [0,0]
            "vfmadd231pd 1192(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"       // C[0:8,0] += A[0:8,0]*B[19,1][0,0]
            "vfmadd231pd 1712(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[19,1][0,12]
            "vfmadd231pd 1760(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[19,1][0,13]
            "vfmadd231pd 2184(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"      // C[0:8,22] += A[0:8,0]*B[19,1][0,22]
            "vfmadd231pd 2208(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[19,1][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=20)
              "vmovapd 1280(%%rdi), %%zmm0\r\n"                           // A [0,20] [0,0]
            "vfmadd231pd 1832(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[20,1][0,14]
            "vfmadd231pd 1944(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[20,1][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=21)
              "vmovapd 1344(%%rdi), %%zmm0\r\n"                           // A [0,21] [0,0]
            "vfmadd231pd 2128(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[21,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=22)
              "vmovapd 1408(%%rdi), %%zmm0\r\n"                           // A [0,22] [0,0]
            "vfmadd231pd 1248(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[22,1][0,1]
            "vfmadd231pd 1552(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[22,1][0,8]
            "vfmadd231pd 1840(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[22,1][0,14]
            "vfmadd231pd 1952(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[22,1][0,17]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=24)
              "vmovapd 1536(%%rdi), %%zmm0\r\n"                           // A [0,24] [0,0]
            "vfmadd231pd 1200(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"       // C[0:8,0] += A[0:8,0]*B[24,1][0,0]
            "vfmadd231pd 1624(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[24,1][0,10]
            "vfmadd231pd 1976(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[24,1][0,18]
            "vfmadd231pd 2056(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[24,1][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=25)
              "vmovapd 1600(%%rdi), %%zmm0\r\n"                           // A [0,25] [0,0]
            "vfmadd231pd 1448(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[25,1][0,6]
            "vfmadd231pd 1632(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[25,1][0,10]
            "vfmadd231pd 1720(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[25,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=26)
              "vmovapd 1664(%%rdi), %%zmm0\r\n"                           // A [0,26] [0,0]
            "vfmadd231pd 1344(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"       // C[0:8,3] += A[0:8,0]*B[26,1][0,3]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=27)
              "vmovapd 1728(%%rdi), %%zmm0\r\n"                           // A [0,27] [0,0]
            "vfmadd231pd 1560(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[27,1][0,8]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=28)
              "vmovapd 1792(%%rdi), %%zmm0\r\n"                           // A [0,28] [0,0]
            "vfmadd231pd 1400(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"       // C[0:8,5] += A[0:8,0]*B[28,1][0,5]
            "vfmadd231pd 1848(%%rsi)%{1to8%}, %%zmm0, %%zmm18\r\n"      // C[0:8,14] += A[0:8,0]*B[28,1][0,14]
            "vfmadd231pd 2232(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[28,1][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=29)
              "vmovapd 1856(%%rdi), %%zmm0\r\n"                           // A [0,29] [0,0]
            "vfmadd231pd 1312(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[29,1][0,2]
            "vfmadd231pd 1408(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"       // C[0:8,5] += A[0:8,0]*B[29,1][0,5]
            "vfmadd231pd 1640(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[29,1][0,10]
            "vfmadd231pd 1904(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"      // C[0:8,16] += A[0:8,0]*B[29,1][0,16]
            "vfmadd231pd 2136(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[29,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=30)
              "vmovapd 1920(%%rdi), %%zmm0\r\n"                           // A [0,30] [0,0]
            "vfmadd231pd 1984(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[30,1][0,18]
            "vfmadd231pd 2144(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[30,1][0,21]
            "vfmadd231pd 2240(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[30,1][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=31)
              "vmovapd 1984(%%rdi), %%zmm0\r\n"                           // A [0,31] [0,0]
            "vfmadd231pd 1256(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[31,1][0,1]
            "vfmadd231pd 1456(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[31,1][0,6]
            "vfmadd231pd 1992(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[31,1][0,18]
            "vfmadd231pd 2152(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[31,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=33)
              "vmovapd 2112(%%rdi), %%zmm0\r\n"                           // A [0,33] [0,0]
            "vfmadd231pd 1320(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[33,1][0,2]
            "vfmadd231pd 1464(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[33,1][0,6]
            "vfmadd231pd 2000(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[33,1][0,18]
            "vfmadd231pd 2328(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[33,1][0,26]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=34)
              "vmovapd 2176(%%rdi), %%zmm0\r\n"                           // A [0,34] [0,0]
            "vfmadd231pd 1352(%%rsi)%{1to8%}, %%zmm0, %%zmm7\r\n"       // C[0:8,3] += A[0:8,0]*B[34,1][0,3]
            "vfmadd231pd 1504(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"      // C[0:8,7] += A[0:8,0]*B[34,1][0,7]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=35)
              "vmovapd 2240(%%rdi), %%zmm0\r\n"                           // A [0,35] [0,0]
            "vfmadd231pd 1728(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[35,1][0,12]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=37)
              "vmovapd 2368(%%rdi), %%zmm0\r\n"                           // A [0,37] [0,0]
            "vfmadd231pd 1328(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[37,1][0,2]
            "vfmadd231pd 2064(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[37,1][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=38)
              "vmovapd 2432(%%rdi), %%zmm0\r\n"                           // A [0,38] [0,0]
            "vfmadd231pd 1416(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"       // C[0:8,5] += A[0:8,0]*B[38,1][0,5]
            "vfmadd231pd 1472(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[38,1][0,6]
            "vfmadd231pd 1960(%%rsi)%{1to8%}, %%zmm0, %%zmm21\r\n"      // C[0:8,17] += A[0:8,0]*B[38,1][0,17]
            "vfmadd231pd 2072(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[38,1][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=39)
              "vmovapd 2496(%%rdi), %%zmm0\r\n"                           // A [0,39] [0,0]
            "vfmadd231pd 1664(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"      // C[0:8,11] += A[0:8,0]*B[39,1][0,11]
            "vfmadd231pd 2192(%%rsi)%{1to8%}, %%zmm0, %%zmm26\r\n"      // C[0:8,22] += A[0:8,0]*B[39,1][0,22]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=40)
              "vmovapd 2560(%%rdi), %%zmm0\r\n"                           // A [0,40] [0,0]
            "vfmadd231pd 1648(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[40,1][0,10]
            "vfmadd231pd 1672(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"      // C[0:8,11] += A[0:8,0]*B[40,1][0,11]
            "vfmadd231pd 2248(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[40,1][0,24]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=42)
              "vmovapd 2688(%%rdi), %%zmm0\r\n"                           // A [0,42] [0,0]
            "vfmadd231pd 1424(%%rsi)%{1to8%}, %%zmm0, %%zmm9\r\n"       // C[0:8,5] += A[0:8,0]*B[42,1][0,5]
            "vfmadd231pd 1480(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[42,1][0,6]
            "vfmadd231pd 1768(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[42,1][0,13]
            "vfmadd231pd 1872(%%rsi)%{1to8%}, %%zmm0, %%zmm19\r\n"      // C[0:8,15] += A[0:8,0]*B[42,1][0,15]
            "vfmadd231pd 2160(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[42,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=43)
              "vmovapd 2752(%%rdi), %%zmm0\r\n"                           // A [0,43] [0,0]
            "vfmadd231pd 1680(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"      // C[0:8,11] += A[0:8,0]*B[43,1][0,11]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=44)
              "vmovapd 2816(%%rdi), %%zmm0\r\n"                           // A [0,44] [0,0]
            "vfmadd231pd 1208(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"       // C[0:8,0] += A[0:8,0]*B[44,1][0,0]
            "vfmadd231pd 1488(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[44,1][0,6]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=45)
              "vmovapd 2880(%%rdi), %%zmm0\r\n"                           // A [0,45] [0,0]
            "vfmadd231pd 1216(%%rsi)%{1to8%}, %%zmm0, %%zmm4\r\n"       // C[0:8,0] += A[0:8,0]*B[45,1][0,0]
            "vfmadd231pd 1912(%%rsi)%{1to8%}, %%zmm0, %%zmm20\r\n"      // C[0:8,16] += A[0:8,0]*B[45,1][0,16]
            "vfmadd231pd 2256(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[45,1][0,24]
            "vfmadd231pd 2304(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[45,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=46)
              "vmovapd 2944(%%rdi), %%zmm0\r\n"                           // A [0,46] [0,0]
            "vfmadd231pd 1264(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[46,1][0,1]
            "vfmadd231pd 1776(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[46,1][0,13]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=47)
              "vmovapd 3008(%%rdi), %%zmm0\r\n"                           // A [0,47] [0,0]
            "vfmadd231pd 2216(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[47,1][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=48)
              "vmovapd 3072(%%rdi), %%zmm0\r\n"                           // A [0,48] [0,0]
            "vfmadd231pd 1496(%%rsi)%{1to8%}, %%zmm0, %%zmm10\r\n"      // C[0:8,6] += A[0:8,0]*B[48,1][0,6]
            "vfmadd231pd 1656(%%rsi)%{1to8%}, %%zmm0, %%zmm14\r\n"      // C[0:8,10] += A[0:8,0]*B[48,1][0,10]
            "vfmadd231pd 2080(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[48,1][0,20]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=49)
              "vmovapd 3136(%%rdi), %%zmm0\r\n"                           // A [0,49] [0,0]
            "vfmadd231pd 1336(%%rsi)%{1to8%}, %%zmm0, %%zmm6\r\n"       // C[0:8,2] += A[0:8,0]*B[49,1][0,2]
            "vfmadd231pd 1376(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"       // C[0:8,4] += A[0:8,0]*B[49,1][0,4]
            "vfmadd231pd 1784(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[49,1][0,13]
            "vfmadd231pd 2008(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[49,1][0,18]
            "vfmadd231pd 2312(%%rsi)%{1to8%}, %%zmm0, %%zmm29\r\n"      // C[0:8,25] += A[0:8,0]*B[49,1][0,25]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=50)
              "vmovapd 3200(%%rdi), %%zmm0\r\n"                           // A [0,50] [0,0]
            "vfmadd231pd 1384(%%rsi)%{1to8%}, %%zmm0, %%zmm8\r\n"       // C[0:8,4] += A[0:8,0]*B[50,1][0,4]
            "vfmadd231pd 1512(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"      // C[0:8,7] += A[0:8,0]*B[50,1][0,7]
            "vfmadd231pd 1608(%%rsi)%{1to8%}, %%zmm0, %%zmm13\r\n"      // C[0:8,9] += A[0:8,0]*B[50,1][0,9]
            "vfmadd231pd 2224(%%rsi)%{1to8%}, %%zmm0, %%zmm27\r\n"      // C[0:8,23] += A[0:8,0]*B[50,1][0,23]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=51)
              "vmovapd 3264(%%rdi), %%zmm0\r\n"                           // A [0,51] [0,0]
            "vfmadd231pd 1272(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[51,1][0,1]
            "vfmadd231pd 1688(%%rsi)%{1to8%}, %%zmm0, %%zmm15\r\n"      // C[0:8,11] += A[0:8,0]*B[51,1][0,11]
            "vfmadd231pd 1736(%%rsi)%{1to8%}, %%zmm0, %%zmm16\r\n"      // C[0:8,12] += A[0:8,0]*B[51,1][0,12]
            "vfmadd231pd 2048(%%rsi)%{1to8%}, %%zmm0, %%zmm23\r\n"      // C[0:8,19] += A[0:8,0]*B[51,1][0,19]
            "vfmadd231pd 2336(%%rsi)%{1to8%}, %%zmm0, %%zmm30\r\n"      // C[0:8,26] += A[0:8,0]*B[51,1][0,26]
            "vfmadd231pd 2344(%%rsi)%{1to8%}, %%zmm0, %%zmm31\r\n"      // C[0:8,27] += A[0:8,0]*B[51,1][0,27]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=52)
              "vmovapd 3328(%%rdi), %%zmm0\r\n"                           // A [0,52] [0,0]
            "vfmadd231pd 1520(%%rsi)%{1to8%}, %%zmm0, %%zmm11\r\n"      // C[0:8,7] += A[0:8,0]*B[52,1][0,7]
            "vfmadd231pd 2016(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[52,1][0,18]
            "vfmadd231pd 2088(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[52,1][0,20]
            "vfmadd231pd 2168(%%rsi)%{1to8%}, %%zmm0, %%zmm25\r\n"      // C[0:8,21] += A[0:8,0]*B[52,1][0,21]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=53)
              "vmovapd 3392(%%rdi), %%zmm0\r\n"                           // A [0,53] [0,0]
            "vfmadd231pd 1568(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[53,1][0,8]
            "vfmadd231pd 1792(%%rsi)%{1to8%}, %%zmm0, %%zmm17\r\n"      // C[0:8,13] += A[0:8,0]*B[53,1][0,13]
            "vfmadd231pd 2024(%%rsi)%{1to8%}, %%zmm0, %%zmm22\r\n"      // C[0:8,18] += A[0:8,0]*B[53,1][0,18]
            // Block GEMM microkernel
              // Load A register block @ (d=0,r=54)
              "vmovapd 3456(%%rdi), %%zmm0\r\n"                           // A [0,54] [0,0]
            "vfmadd231pd 1280(%%rsi)%{1to8%}, %%zmm0, %%zmm5\r\n"       // C[0:8,1] += A[0:8,0]*B[54,1][0,1]
            "vfmadd231pd 1576(%%rsi)%{1to8%}, %%zmm0, %%zmm12\r\n"      // C[0:8,8] += A[0:8,0]*B[54,1][0,8]
            "vfmadd231pd 2096(%%rsi)%{1to8%}, %%zmm0, %%zmm24\r\n"      // C[0:8,20] += A[0:8,0]*B[54,1][0,20]
            "vfmadd231pd 2264(%%rsi)%{1to8%}, %%zmm0, %%zmm28\r\n"      // C[0:8,24] += A[0:8,0]*B[54,1][0,24]
            // Store C register block @ (d=0,r=0)
            "vmovapd %%zmm4, 0(%%rdx)\r\n"                              // C [0,0] [0,0]
            "vmovapd %%zmm5, 64(%%rdx)\r\n"                             // C [0,0] [0,1]
            "vmovapd %%zmm6, 128(%%rdx)\r\n"                            // C [0,0] [0,2]
            "vmovapd %%zmm7, 192(%%rdx)\r\n"                            // C [0,0] [0,3]
            "vmovapd %%zmm8, 256(%%rdx)\r\n"                            // C [0,0] [0,4]
            "vmovapd %%zmm9, 320(%%rdx)\r\n"                            // C [0,0] [0,5]
            "vmovapd %%zmm10, 384(%%rdx)\r\n"                           // C [0,0] [0,6]
            "vmovapd %%zmm11, 448(%%rdx)\r\n"                           // C [0,0] [0,7]
            "vmovapd %%zmm12, 512(%%rdx)\r\n"                           // C [0,0] [0,8]
            "vmovapd %%zmm13, 576(%%rdx)\r\n"                           // C [0,0] [0,9]
            "vmovapd %%zmm14, 640(%%rdx)\r\n"                           // C [0,0] [0,10]
            "vmovapd %%zmm15, 704(%%rdx)\r\n"                           // C [0,0] [0,11]
            "vmovapd %%zmm16, 768(%%rdx)\r\n"                           // C [0,0] [0,12]
            "vmovapd %%zmm17, 832(%%rdx)\r\n"                           // C [0,0] [0,13]
            "vmovapd %%zmm18, 896(%%rdx)\r\n"                           // C [0,0] [0,14]
            "vmovapd %%zmm19, 960(%%rdx)\r\n"                           // C [0,0] [0,15]
            "vmovapd %%zmm20, 1024(%%rdx)\r\n"                          // C [0,0] [0,16]
            "vmovapd %%zmm21, 1088(%%rdx)\r\n"                          // C [0,0] [0,17]
            "vmovapd %%zmm22, 1152(%%rdx)\r\n"                          // C [0,0] [0,18]
            "vmovapd %%zmm23, 1216(%%rdx)\r\n"                          // C [0,0] [0,19]
            "vmovapd %%zmm24, 1280(%%rdx)\r\n"                          // C [0,0] [0,20]
            "vmovapd %%zmm25, 1344(%%rdx)\r\n"                          // C [0,0] [0,21]
            "vmovapd %%zmm26, 1408(%%rdx)\r\n"                          // C [0,0] [0,22]
            "vmovapd %%zmm27, 1472(%%rdx)\r\n"                          // C [0,0] [0,23]
            "vmovapd %%zmm28, 1536(%%rdx)\r\n"                          // C [0,0] [0,24]
            "vmovapd %%zmm29, 1600(%%rdx)\r\n"                          // C [0,0] [0,25]
            "vmovapd %%zmm30, 1664(%%rdx)\r\n"                          // C [0,0] [0,26]
            "vmovapd %%zmm31, 1728(%%rdx)\r\n"                          // C [0,0] [0,27]
        "addq $64, %%rdi\r\n"                                       // Move A to (d=1,r=0)
        "addq $-1728, %%rdx\r\n"                                    // Move C to (d=1,r=-1)
        "addq $1, %%r12\r\n"
        "cmp $1, %%r12\r\n"
        "jl LOOP_TOP_0_%=\r\n"

    : : "m"(A), "m"(B), "m"(C) : "r12","rdi","rdx","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9");

};
