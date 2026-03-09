      PROGRAM HEAVYCOMB
      IMPLICIT NONE
      INTEGER I, J
      REAL A, B, C, S, T, X, Y, UNUSED
      A = 2.5
      B = 3.7
      C = 1.2
      S = 0.0
      DO I = 1, 5000
          DO J = 1, 5000
              T = A * B + C
              X = 7.0 * 8.0 + 3.0 * 2.0
              Y = FLOAT(I) * 0.001 + FLOAT(J) * 0.0001
              UNUSED = 99.9 * 88.8 + 77.7
              S = S + T + X + Y
          ENDDO
      ENDDO
      PRINT *, S
      END
