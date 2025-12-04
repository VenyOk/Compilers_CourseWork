      PROGRAM IFTEST
      IMPLICIT NONE
      INTEGER X, Y
      X = 10
      IF (X .GT. 5) THEN
          Y = 1
      ELSE
          Y = 0
      ENDIF
      PRINT *, Y
      END

