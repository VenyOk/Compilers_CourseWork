      PROGRAM COMPOP
      IMPLICIT NONE
      INTEGER A, B
      LOGICAL L1, L2, L3, L4
      A = 5
      B = 10
      L1 = A .EQ. B
      L2 = A .NE. B
      L3 = A .LT. B
      L4 = A .GE. B
      PRINT *, L1, L2, L3, L4
      END

