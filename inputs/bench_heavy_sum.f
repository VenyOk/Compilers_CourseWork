      PROGRAM HSUM
      IMPLICIT NONE
      INTEGER I, J
      REAL S, A, B, C, T
      A = 1.5
      B = 2.7
      C = 3.14
      S = 0.0
      DO I = 1, 5000
          DO J = 1, 5000
              T = (A * B + C) / (A - B + C)
              S = S + T + FLOAT(I) * 0.0001 + FLOAT(J) * 0.00001
          ENDDO
      ENDDO
      PRINT *, S
      END
