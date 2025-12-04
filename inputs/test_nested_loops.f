      PROGRAM NESTED
      IMPLICIT NONE
      INTEGER I, J, SUM
      SUM = 0
      DO I = 1, 5
          DO J = 1, 3
              SUM = SUM + I * J
          ENDDO
      ENDDO
      PRINT *, SUM
      END

