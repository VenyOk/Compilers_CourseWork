      PROGRAM CPROP
      IMPLICIT NONE
      INTEGER I, N, M, K, S
      REAL A, B, C, R
      N = 10
      M = 20
      K = N * M
      S = 0
      DO I = 1, 1000
          S = S + K
          S = S + N + M
          S = S + N * 3 + M * 2
      ENDDO
      PRINT *, S
      A = 3.14
      B = 2.71
      C = A * B
      R = 0.0
      DO I = 1, 1000
          R = R + C
          R = R + A + B
          R = R + A * 2.0 + B * 3.0
      ENDDO
      PRINT *, R
      END
