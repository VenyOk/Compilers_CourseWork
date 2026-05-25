      PROGRAM SAXPY
      IMPLICIT NONE
      INTEGER N, I
      REAL X(500000), Y(500000), A
      N = 500000
      A = 2.5
      DO I = 1, N
          X(I) = FLOAT(MOD(I * 13, 997)) / 997.0
          Y(I) = FLOAT(MOD(I * 29, 991)) / 991.0
      ENDDO
      DO I = 1, N
          Y(I) = A * X(I) + Y(I)
      ENDDO
      PRINT *, Y(1)
      PRINT *, Y(N)
      PRINT *, Y(N / 2)
      END
