      PROGRAM DOTPRD
      IMPLICIT NONE
      INTEGER N, I
      REAL X(500000), Y(500000), S, A
      N = 500000
      A = 0.001
      DO I = 1, N
          X(I) = FLOAT(MOD(I * 17, 997)) * A
          Y(I) = FLOAT(MOD(I * 23, 991)) * A
      ENDDO
      S = 0.0
      DO I = 1, N
          S = S + X(I) * Y(I)
      ENDDO
      PRINT *, S
      END
