      PROGRAM NESTCT
      IMPLICIT NONE
      INTEGER I, J, SUM, COUNT
      SUM = 0
      COUNT = 0
      DO I = 1, 10
          IF (I .GT. 5) THEN
              DO J = 1, I
                  IF (J .LT. 3) THEN
                      SUM = SUM + I * J
                      COUNT = COUNT + 1
                  ELSE
                      SUM = SUM - J
                  ENDIF
              ENDDO
          ELSE
              SUM = SUM + I
          ENDIF
      ENDDO
      PRINT *, SUM
      PRINT *, COUNT
      END

