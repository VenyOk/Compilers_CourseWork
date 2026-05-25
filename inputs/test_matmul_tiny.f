      PROGRAM T
      IMPLICIT NONE
      INTEGER N, I, J, K, SEED
      REAL A(128, 128), B(128, 128), C(128, 128), S
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
      PRINT *, C(1, 1)
      PRINT *, C(65, 65)
      PRINT *, C(32, 33)
      END
