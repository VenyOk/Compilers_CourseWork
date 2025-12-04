      PROGRAM MULTSU
      IMPLICIT NONE
      INTEGER X, Y, Z
      X = 10
      Y = 20
      Z = 5
      CALL ADD(X, Y, Z)
      PRINT *, X, Y, Z
      CALL MULTIP(X, Y)
      PRINT *, X, Y
      END

      SUBROUTINE ADD(A, B, C)
      INTEGER A, B, C
      A = A + 1
      B = B + 2
      C = C + 3
      END

      SUBROUTINE MULTIP(A, B)
      INTEGER A, B, TEMP
      TEMP = A
      A = A * B
      B = TEMP * B
      END

