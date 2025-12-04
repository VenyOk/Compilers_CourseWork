      PROGRAM CMPLXE
      IMPLICIT NONE
      INTEGER A, B, C, D, RESULT
      REAL X, Y, Z
      LOGICAL L1, L2, L3
      A = 10
      B = 5
      C = 3
      D = 2
      X = 2.5
      Y = 1.5
      RESULT = (A + B) * C - D / 2
      Z = X * Y + (A - B) / 2.0
      L1 = A .GT. B .AND. C .LT. D
      L2 = (A + B) .EQ. (C * D) .OR. (A .GT. 20)
      L3 = .NOT. L1 .AND. L2
      PRINT *, RESULT
      PRINT *, Z
      PRINT *, L1, L2, L3
      END

