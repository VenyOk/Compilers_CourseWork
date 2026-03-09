      PROGRAM BSTR
      IMPLICIT NONE
      INTEGER I, S1, S2, S3
      REAL X, R1, R2
      S1 = 0
      S2 = 0
      S3 = 0
      DO I = 1, 1000
          S1 = S1 + I * 2
          S2 = S2 + I * 4
          S3 = S3 + I * 8
      ENDDO
      PRINT *, S1
      PRINT *, S2
      PRINT *, S3
      R1 = 0.0
      R2 = 0.0
      DO I = 1, 1000
          X = FLOAT(I)
          R1 = R1 + X ** 2
          R2 = R2 + X * 2.0
      ENDDO
      PRINT *, R1
      PRINT *, R2
      END
