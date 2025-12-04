      PROGRAM EQVN
      IMPLICIT NONE
      LOGICAL L1, L2, L3, L4
      L1 = .TRUE.
      L2 = .FALSE.
      L3 = L1 .EQV. L2
      L4 = L1 .NEQV. L2
      PRINT *, L3, L4
      END

