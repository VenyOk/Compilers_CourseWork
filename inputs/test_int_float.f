      PROGRAM INTF
      IMPLICIT NONE
      INTEGER A, B
      REAL X, Y
      X = 3.7
      Y = 2.3
      A = INT(X)
      B = INT(Y)
      X = FLOAT(A)
      Y = FLOAT(B)
      PRINT *, A, B
      PRINT *, X, Y
      END

