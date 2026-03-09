      PROGRAM BCSE
      IMPLICIT NONE
      INTEGER I, J
      REAL A, B, C, S1, S2, S3
      A = 3.14
      B = 2.71
      C = 1.41
      S1 = 0.0
      S2 = 0.0
      S3 = 0.0
      DO I = 1, 1000
          S1 = S1 + (A * B + C)
          S2 = S2 + (A * B + C) * 2.0
          S3 = S3 + (A * B + C) / 3.0
      ENDDO
      PRINT *, S1
      PRINT *, S2
      PRINT *, S3
      DO I = 1, 100
          DO J = 1, 100
              S1 = S1 + (A + B) * (A - B)
              S2 = S2 + (A + B) * C
              S3 = S3 + (A - B) * C
          ENDDO
      ENDDO
      PRINT *, S1
      END
