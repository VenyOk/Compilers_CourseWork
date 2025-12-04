      PROGRAM NEST
      IMPLICIT NONE
      INTEGER A(2, 3, 2)
      INTEGER I, J, K, VAL
      VAL = 1
      DO I = 1, 2
          DO J = 1, 3
              DO K = 1, 2
                  A(I, J, K) = VAL
                  VAL = VAL + 1
              ENDDO
          ENDDO
      ENDDO
      PRINT *, A(1, 1, 1)
      PRINT *, A(2, 3, 2)
      END

