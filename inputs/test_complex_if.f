      PROGRAM CPLXIF
      IMPLICIT NONE
      INTEGER X, Y, RESULT
      X = 10
      Y = 20
      IF (X .GT. 5) THEN
          IF (Y .GT. 15) THEN
              RESULT = X + Y
          ELSE
              RESULT = X - Y
          ENDIF
      ELSE
          RESULT = 0
      ENDIF
      PRINT *, RESULT
      END

