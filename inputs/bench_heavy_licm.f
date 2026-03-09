      PROGRAM HEAVYLICM
      IMPLICIT NONE
      INTEGER I
      REAL A, B, C, D, T1, T2, S
      A = 3.14
      B = 2.71
      C = 1.41
      D = 9.81
      S = 0.0
      DO I = 1, 10000000
          T1 = A * B + C
          T2 = C * D - A
          S = S + T1 + T2 + FLOAT(I) * 0.000001
      ENDDO
      PRINT *, S
      END
