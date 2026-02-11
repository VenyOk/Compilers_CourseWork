      PROGRAM MTRANS
      IMPLICIT NONE
      INTEGER A(3, 3), B(3, 3), I, J
      INTEGER VAL
      VAL = 1
      DO I = 1, 3
          DO J = 1, 3
              A(I, J) = VAL
              VAL = VAL + 1
          ENDDO
      ENDDO
      CALL TRANSP(A, B)
      DO I = 1, 3
          DO J = 1, 3
              PRINT *, B(I, J)
          ENDDO
      ENDDO
      END

      SUBROUTINE TRANSP(A, B)
      IMPLICIT NONE
      INTEGER A(3, 3), B(3, 3), I, J
      DO I = 1, 3
          DO J = 1, 3
              B(I, J) = A(J, I)
          ENDDO
      ENDDO
      END
