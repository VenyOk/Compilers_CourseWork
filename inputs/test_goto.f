      PROGRAM GOT
      IMPLICIT NONE
      INTEGER I, SUM
      SUM = 0
      I = 1
    10 CONTINUE
      SUM = SUM + I
      I = I + 1
      IF (I .LE. 10) GOTO 10
      PRINT *, SUM
      END

