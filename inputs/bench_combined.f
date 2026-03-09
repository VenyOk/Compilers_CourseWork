      PROGRAM BCOMB
      IMPLICIT NONE
      INTEGER I, J, K, N
      REAL A(10, 10), B(10, 10), C(10, 10)
      REAL SCALE, OFFSET, TMP, TOTAL
      N = 10
      SCALE = 2.5
      OFFSET = 1.0 + 0.5
      DO I = 1, N
          DO J = 1, N
              A(I, J) = FLOAT(I) * SCALE + OFFSET
              B(I, J) = FLOAT(J) * SCALE - OFFSET
              C(I, J) = 0.0
          ENDDO
      ENDDO
      DO I = 1, N
          DO J = 1, N
              TMP = 0.0
              DO K = 1, N
                  TMP = TMP + A(I, K) * B(K, J)
              ENDDO
              C(I, J) = TMP * SCALE + OFFSET
          ENDDO
      ENDDO
      TOTAL = 0.0
      DO I = 1, N
          DO J = 1, N
              TOTAL = TOTAL + C(I, J)
          ENDDO
      ENDDO
      PRINT *, TOTAL
      END
