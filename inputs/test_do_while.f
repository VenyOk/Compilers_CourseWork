      PROGRAM DOWHIL
      IMPLICIT NONE
      INTEGER I, SUM
      SUM = 0
      I = 1
      DO WHILE (I .LE. 10)
          SUM = SUM + I
          I = I + 1
      ENDDO
      PRINT *, SUM
      END

