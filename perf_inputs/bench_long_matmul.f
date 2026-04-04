      PROGRAM LMMUL
      IMPLICIT NONE
      INTEGER N, I, J, K
      REAL A(640,640), B(640,640), C(640,640)
      N = 640
      DO I = 1, N
          DO J = 1, N
              A(I,J) = FLOAT(I + J) / 1280.0
              B(I,J) = FLOAT(I - J + N) / 1280.0
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
