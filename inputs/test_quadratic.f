      PROGRAM QUADR
      IMPLICIT NONE
      REAL A, B, C, D, X1, X2
      A = 1.0
      B = -5.0
      C = 6.0
      D = B * B - 4.0 * A * C
      IF (D .GT. 0.0) THEN
          X1 = (-B + SQRT(D)) / (2.0 * A)
          X2 = (-B - SQRT(D)) / (2.0 * A)
          PRINT *, X1
          PRINT *, X2
      ELSEIF (D .EQ. 0.0) THEN
          X1 = -B / (2.0 * A)
          PRINT *, X1
      ELSE
          PRINT *, -1
      ENDIF
      A = 1.0
      B = -2.0
      C = 1.0
      D = B * B - 4.0 * A * C
      IF (D .GT. 0.0) THEN
          X1 = (-B + SQRT(D)) / (2.0 * A)
          X2 = (-B - SQRT(D)) / (2.0 * A)
          PRINT *, X1
          PRINT *, X2
      ELSEIF (D .EQ. 0.0) THEN
          X1 = -B / (2.0 * A)
          PRINT *, X1
      ELSE
          PRINT *, -1
      ENDIF
      END
