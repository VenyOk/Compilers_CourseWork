      PROGRAM CFOLD
      IMPLICIT NONE
      INTEGER I, S
      REAL X, Y, Z
      S = 0
      DO I = 1, 1000
          S = S + (3 + 4) * (10 - 2)
          S = S + 100 / 5 + 2 * 3
          S = S - (7 * 8 - 6 * 9)
      ENDDO
      PRINT *, S
      X = 0.0
      DO I = 1, 1000
          X = X + (3.14 * 2.0) / (1.0 + 1.0)
          Y = 100.0 / 4.0 + 25.0 * 0.0
          Z = (2.0 ** 3) * (4.0 / 2.0)
          X = X + Y + Z
      ENDDO
      PRINT *, X
      END
