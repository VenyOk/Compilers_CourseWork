      PROGRAM MIXTYP
      IMPLICIT NONE
      INTEGER A, B
      REAL X, Y, Z
      LOGICAL L
      A = 7
      B = 3
      X = 2.5
      Y = 1.5
      Z = A + X
      Z = Z * B - Y
      L = (A + B) .GT. 10 .AND. X .LT. 5.0
      IF (L) THEN
          Z = Z + 10.0
      ELSE
          Z = Z - 5.0
      ENDIF
      PRINT *, Z
      PRINT *, L
      END

