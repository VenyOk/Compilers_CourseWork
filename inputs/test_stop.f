      PROGRAM STP
      IMPLICIT NONE
      INTEGER X
      X = 10
      IF (X .GT. 5) THEN
          PRINT *, X
          STOP
      ENDIF
      PRINT *, 0
      END

