      PROGRAM FACTR
      IMPLICIT NONE
      INTEGER I, N, FACT
      N = 10
      FACT = 1
      DO I = 2, N
          FACT = FACT * I
      ENDDO
      PRINT *, FACT
      N = 12
      FACT = 1
      DO I = 2, N
          FACT = FACT * I
      ENDDO
      PRINT *, FACT
      END
