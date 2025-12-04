      PROGRAM SUBTES
      IMPLICIT NONE
      INTEGER X, Y
      X = 5
      Y = 10
      CALL SWAP(X, Y)
      PRINT *, X, Y
      END

      SUBROUTINE SWAP(A, B)
      INTEGER A, B, TEMP
      TEMP = A
      A = B
      B = TEMP
      END

