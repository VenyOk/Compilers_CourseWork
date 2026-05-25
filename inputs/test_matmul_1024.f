      PROGRAM T
      IMPLICIT NONE
      INTEGER N, I, J, K, SEED
      REAL A(1024, 1024), B(1024, 1024), C(1024, 1024), S
      N = 65
      SEED = 12345
      DO I = 1, N
          DO J = 1, N
              A(I,J) = FLOAT(MOD(ABS((I*17+J*23+SEED)*31), 10))
          ENDDO
      ENDDO
      SEED = SEED + 1000
      DO I = 1, N
          DO J = 1, N
              B(I,J) = FLOAT(MOD(ABS((I*17+J*23+SEED)*31), 10))
          ENDDO
      ENDDO
      DO I = 1, N
          DO J = 1, N
              S = 0.0
              DO K = 1, N
                  S = S + A(I, K) * B(K, J)
              ENDDO
              C(I, J) = S
          ENDDO
      ENDDO
      CALL CHKSUM(C, N, 1024, S)
      PRINT *, S
      END

      SUBROUTINE CHKSUM(MAT, N, MAXN, S)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL MAT(1024, 1024)
      REAL S
      INTEGER I, J
      S = 0.0
      DO I = 1, N
          DO J = 1, N
              S = S + MAT(I, J)
          ENDDO
      ENDDO
      END
