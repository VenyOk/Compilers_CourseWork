      PROGRAM CONT
      IMPLICIT NONE
      INTEGER I, SUM
      SUM = 0
      DO I = 1, 10
          IF (I .EQ. 5) THEN
              CONTINUE
          ELSE
              SUM = SUM + I
          ENDIF
      ENDDO
      PRINT *, SUM
      END

