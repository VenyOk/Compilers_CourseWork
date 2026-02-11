      PROGRAM DBLPRC
      IMPLICIT NONE
      REAL A, B, C, D, PI
      REAL SUM
      INTEGER I
      PI = 3.14159265358979
      PRINT *, PI
      A = 1.23456789012345
      B = 9.87654321098765
      C = A + B
      PRINT *, C
      A = 0.000001
      B = 0.000001
      D = A * B
      PRINT *, D
      A = 1.0
      B = 3.0
      C = A / B
      PRINT *, C
      SUM = 0.0
      DO I = 1, 100
          SUM = SUM + 1.0 / I
      ENDDO
      PRINT *, SUM
      A = 2.0
      B = SQRT(A)
      PRINT *, B
      A = 2.0
      B = 10.0
      C = A ** B
      PRINT *, C
      END
