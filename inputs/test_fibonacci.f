      PROGRAM FIBON
      IMPLICIT NONE
      INTEGER N, I, FIB1, FIB2, FIB3
      N = 10
      FIB1 = 0
      FIB2 = 1
      PRINT *, FIB1
      PRINT *, FIB2
      DO I = 3, N
          FIB3 = FIB1 + FIB2
          FIB1 = FIB2
          FIB2 = FIB3
          PRINT *, FIB3
      ENDDO
      END

