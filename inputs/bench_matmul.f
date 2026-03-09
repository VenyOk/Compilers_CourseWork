      PROGRAM MATMUL
      IMPLICIT NONE
      INTEGER N, I, J, K
      REAL A(100,100), B(100,100), C(100,100)
      N = 100
      DO I = 1, N
        DO J = 1, N
          A(I,J) = FLOAT(I + J) / 200.0
          B(I,J) = FLOAT(I - J + 100) / 200.0
          C(I,J) = 0.0
        ENDDO
      ENDDO
      DO I = 1, N
        DO J = 1, N
          DO K = 1, N
            C(I,J) = C(I,J) + A(I,K) * B(K,J)
          ENDDO
        ENDDO
      ENDDO
      WRITE(*,*) C(1,1), C(N,N)
      END
