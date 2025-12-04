      PROGRAM MATHFN
      IMPLICIT NONE
      REAL X, Y, Z, W
      X = 4.0
      Y = SQRT(X)
      Z = SIN(X) + COS(X)
      W = EXP(LOG(X)) + POW(X, 2.0)
      PRINT *, Y
      PRINT *, Z
      PRINT *, W
      END

