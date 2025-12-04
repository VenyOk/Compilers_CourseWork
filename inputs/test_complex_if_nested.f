      PROGRAM CPLXI
      IMPLICIT NONE
      INTEGER X, Y, Z
      X = 10
      Y = 20
      Z = 5
      IF (X .GT. 5) THEN
          IF (Y .GT. 15) THEN
              IF (Z .LT. 10) THEN
                  PRINT *, 1
              ELSE
                  PRINT *, 2
              ENDIF
          ELSE
              PRINT *, 3
          ENDIF
      ELSE
          PRINT *, 4
      ENDIF
      END

