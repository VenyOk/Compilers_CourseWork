      PROGRAM DIM
      IMPLICIT NONE
      INTEGER I
      DIMENSION A(10), B(5, 5)
      INTEGER A, B
      DO I = 1, 10
          A(I) = I * 2
      ENDDO
      B(1, 1) = 100
      PRINT *, A(5)
      PRINT *, B(1, 1)
      END

