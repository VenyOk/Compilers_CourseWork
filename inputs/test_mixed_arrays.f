      PROGRAM MIXA
      IMPLICIT NONE
      INTEGER A(5), B(3, 4)
      REAL C(10)
      INTEGER I, J
      DO I = 1, 5
          A(I) = I
      ENDDO
      DO I = 1, 3
          DO J = 1, 4
              B(I, J) = I * J
          ENDDO
      ENDDO
      DO I = 1, 10
          C(I) = I * 0.5
      ENDDO
      PRINT *, A(3)
      PRINT *, B(2, 3)
      PRINT *, C(5)
      END

