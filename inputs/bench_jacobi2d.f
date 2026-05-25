      PROGRAM JCB2D
      IMPLICIT NONE
      INTEGER N, ITERS, I, J, K
      REAL U(256, 256), V(256, 256)
      N = 256
      ITERS = 40
      DO I = 1, N
          DO J = 1, N
              U(I, J) = 0.0
              V(I, J) = 0.0
          ENDDO
      ENDDO
      DO J = 1, N
          U(1, J) = 1.0
          U(N, J) = 1.0
      ENDDO
      DO I = 1, N
          U(I, 1) = 1.0
          U(I, N) = 1.0
      ENDDO
      DO K = 1, ITERS
          DO I = 2, N - 1
              DO J = 2, N - 1
                  V(I, J) = 0.25 * (U(I - 1, J) + U(I + 1, J)
     1                            + U(I, J - 1) + U(I, J + 1))
              ENDDO
          ENDDO
          DO I = 2, N - 1
              DO J = 2, N - 1
                  U(I, J) = V(I, J)
              ENDDO
          ENDDO
      ENDDO
      PRINT *, U(N / 2, N / 2)
      END
