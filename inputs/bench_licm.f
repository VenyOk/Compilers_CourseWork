      PROGRAM BLICM
      IMPLICIT NONE
      INTEGER I, J
      REAL A, B, C, INV, S
      A = 7.5
      B = 3.2
      C = 2.1
      S = 0.0
      DO I = 1, 1000
          S = S + (A * B + C) * FLOAT(I)
      ENDDO
      PRINT *, S
      S = 0.0
      DO I = 1, 100
          DO J = 1, 100
              INV = A / B + C
              S = S + INV * FLOAT(I) + FLOAT(J)
          ENDDO
      ENDDO
      PRINT *, S
      S = 0.0
      DO I = 1, 1000
          S = S + SQRT(A * A + B * B) + FLOAT(I)
      ENDDO
      PRINT *, S
      END
