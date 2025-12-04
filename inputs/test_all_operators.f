      PROGRAM ALLO
      IMPLICIT NONE
      INTEGER A, B, C
      LOGICAL L1, L2, L3
      A = 10
      B = 5
      C = (A + B) * (A - B) / 2
      L1 = A .GT. B .AND. B .LT. A
      L2 = A .EQ. B .OR. A .NE. B
      L3 = .NOT. L1 .EQV. L2
      PRINT *, C
      PRINT *, L1, L2, L3
      END

