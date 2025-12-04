      PROGRAM MATMUL
      IMPLICIT NONE
      INTEGER I, J, K
      INTEGER A(3, 3), B(3, 3), C(3, 3)
      DO I = 1, 3
          DO J = 1, 3
              A(I, J) = I + J
              B(I, J) = I * J
              C(I, J) = 0
          ENDDO
      ENDDO
      DO I = 1, 3
          DO J = 1, 3
              DO K = 1, 3
                  C(I, J) = C(I, J) + A(I, K) * B(K, J)
              ENDDO
          ENDDO
      ENDDO
      PRINT *, C(2, 2)
      END

