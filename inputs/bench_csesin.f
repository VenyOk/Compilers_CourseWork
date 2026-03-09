      PROGRAM CSESIN
      IMPLICIT NONE
      INTEGER I
      REAL S, X
      S = 0.0
      DO I = 1, 1000000
        X = FLOAT(I) * 0.0001
        S = S + SIN(X)*SIN(X) + COS(X)*COS(X) + SIN(X)*COS(X)
      ENDDO
      WRITE(*,*) S
      END
