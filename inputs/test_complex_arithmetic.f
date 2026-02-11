      PROGRAM CMPLXA
      IMPLICIT NONE
      COMPLEX C1, C2, C3, C4, C5
      C1 = (3.0, 4.0)
      C2 = (1.0, -2.0)
      C3 = C1 + C2
      PRINT *, C3
      C4 = C1 - C2
      PRINT *, C4
      C5 = C1 * C2
      PRINT *, C5
      C3 = -C1
      PRINT *, C3
      C3 = C1 + (5.0, 0.0)
      PRINT *, C3
      C3 = (0.0, 1.0)
      C4 = C3 * C3
      PRINT *, C4
      END
