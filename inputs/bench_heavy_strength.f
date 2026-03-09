      PROGRAM HEAVYSTR
      IMPLICIT NONE
      INTEGER I
      REAL X, S
      S = 0.0
      DO I = 1, 10000000
          X = FLOAT(I) * 0.001
          S = S + X ** 2 + X ** 3
      ENDDO
      PRINT *, S
      END
