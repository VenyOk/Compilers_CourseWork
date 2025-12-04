      PROGRAM LOGCOM
      IMPLICIT NONE
      INTEGER A, B, C
      LOGICAL L1, L2, L3, L4, L5
      A = 5
      B = 10
      C = 15
      L1 = A .LT. B .AND. B .LT. C
      L2 = A .GT. B .OR. C .GT. 20
      L3 = .NOT. (A .EQ. B)
      L4 = (A + B) .EQ. C .AND. (B - A) .GT. 0
      L5 = L1 .AND. L2 .OR. L3 .AND. L4
      PRINT *, L1, L2, L3, L4, L5
      END

