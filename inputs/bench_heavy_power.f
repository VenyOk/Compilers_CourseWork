      PROGRAM HEVPOW
      IMPLICIT NONE
      INTEGER I
      REAL X, S, A, B, C
      S = 0.0
      DO I = 1, 5000000
          X = FLOAT(I) * 0.0001
          A = X ** 2
          B = X ** 3
          C = X ** 2 + X ** 3
          S = S + A + B + C
      ENDDO
      PRINT *, S
      END
