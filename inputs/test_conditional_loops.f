      PROGRAM CONDLP
      IMPLICIT NONE
      INTEGER I, SUM, PRODCT
      SUM = 0
      PRODCT = 1
      I = 1
      DO WHILE (I .LE. 10)
          IF (I .GT. 5) THEN
              SUM = SUM + I
          ELSE
              PRODCT = PRODCT * I
          ENDIF
          I = I + 1
      ENDDO
      PRINT *, SUM
      PRINT *, PRODCT
      END

