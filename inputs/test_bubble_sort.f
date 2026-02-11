      PROGRAM BSORT
      IMPLICIT NONE
      INTEGER A(10), N, I, J, TMP
      N = 10
      A(1) = 5
      A(2) = 3
      A(3) = 8
      A(4) = 1
      A(5) = 9
      A(6) = 2
      A(7) = 7
      A(8) = 4
      A(9) = 6
      A(10) = 10
      CALL BSUB(A, N)
      DO I = 1, N
          PRINT *, A(I)
      ENDDO
      END

      SUBROUTINE BSUB(A, N)
      IMPLICIT NONE
      INTEGER A(10), N, I, J, TMP
      DO I = 1, N - 1
          DO J = 1, N - I
              IF (A(J) .GT. A(J + 1)) THEN
                  TMP = A(J)
                  A(J) = A(J + 1)
                  A(J + 1) = TMP
              ENDIF
          ENDDO
      ENDDO
      END
