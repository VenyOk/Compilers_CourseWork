      PROGRAM HEAVYALL
      IMPLICIT NONE
      INTEGER I, J
      REAL A, B, S, T, U, V, W, DEAD
      A = 1.5
      B = 2.5
      S = 0.0
      DO I = 1, 3000
          DO J = 1, 3000
              T = A * B + 1.0
              U = 3.14 * 2.71 + 1.0 * 0.5
              V = FLOAT(I) * 0.001
              W = V ** 2 + V ** 3
              DEAD = 99.0 * 88.0 + 77.0
              S = S + T + U + W
          ENDDO
      ENDDO
      PRINT *, S
      END
