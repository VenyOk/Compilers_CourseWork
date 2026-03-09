      PROGRAM DCODE
      IMPLICIT NONE
      INTEGER I, S, D1, D2, D3, D4, D5
      REAL X, DEAD1, DEAD2, DEAD3
      S = 0
      DO I = 1, 1000
          S = S + I
          D1 = I * 2
          D2 = I * 3
          D3 = D1 + D2
          D4 = D3 * I
          D5 = D4 - D3 + D2 - D1
      ENDDO
      PRINT *, S
      X = 0.0
      DO I = 1, 1000
          X = X + FLOAT(I) * 0.001
          DEAD1 = FLOAT(I) * 3.14
          DEAD2 = DEAD1 / 2.71
          DEAD3 = SQRT(DEAD1) + DEAD2
      ENDDO
      PRINT *, X
      END
