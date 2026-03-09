      PROGRAM HEAVYFOLD
      IMPLICIT NONE
      INTEGER I
      REAL S, X
      S = 0.0
      DO I = 1, 10000000
          X = 3.14 * 2.71 + 1.41 * 1.73 - 0.57 * 0.31
          X = X + 9.81 * 6.67 + 2.99 * 1.38
          S = S + X + FLOAT(I) * 0.000001
      ENDDO
      PRINT *, S
      END
