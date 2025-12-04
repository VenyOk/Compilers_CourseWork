      PROGRAM LOGOPS
      IMPLICIT NONE
      LOGICAL L1, L2, L3
      INTEGER A, B
      A = 5
      B = 10
      L1 = A .LT. B
      L2 = A .GT. B
      L3 = L1 .AND. .NOT. L2
      PRINT *, L3
      END

