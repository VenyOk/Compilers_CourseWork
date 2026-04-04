      PROGRAM LCSIN
      IMPLICIT NONE
      INTEGER I
      REAL S, X
      S = 0.0
      DO I = 1, 20000000
          X = FLOAT(I) * 0.00001
          S = S + SIN(X)*SIN(X) + COS(X)*COS(X) + SIN(X)*COS(X)
      ENDDO
      WRITE(*,*) S
      END
