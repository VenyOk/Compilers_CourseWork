      PROGRAM DOLOOP
      IMPLICIT NONE
      INTEGER I, SUM
      SUM = 0
      DO I = 1, 10
          SUM = SUM + I
      ENDDO
      PRINT *, SUM
      END

