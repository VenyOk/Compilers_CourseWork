      PROGRAM LCINV
      IMPLICIT NONE
      INTEGER I
      REAL S
      S = 0.0
      DO I = 1, 5000000
        S = S + FLOAT(I) * SQRT(3.14159265 * 3.14159265 + 1.0)
        S = S + SQRT(2.71828183 * 2.71828183 - 1.0)
      ENDDO
      WRITE(*,*) S
      END
