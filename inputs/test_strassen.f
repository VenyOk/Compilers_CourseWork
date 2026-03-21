      PROGRAM STRASS
      IMPLICIT NONE
      INTEGER MAXN
      PARAMETER (MAXN = 1024)
      INTEGER N
      REAL A(1024, 1024)
      REAL B(1024, 1024)
      REAL C(1024, 1024)
      REAL S
      INTEGER SEED
      
      N = 8
      
      SEED = 12345
      CALL GENRND(A, N, MAXN, SEED)
      SEED = SEED + 1000
      CALL GENRND(B, N, MAXN, SEED)
      
      CALL STRASN(A, B, C, N, MAXN)
      
      IF (N .LE. 8) THEN
          PRINT *, 'Matrix A:'
          CALL PRNTMT(A, N, MAXN)
          PRINT *, 'Matrix B:'
          CALL PRNTMT(B, N, MAXN)
          PRINT *, 'Result C = A * B:'
          CALL PRNTMT(C, N, MAXN)
      ENDIF
      
      CALL CHKSUM(C, N, MAXN, S)
      PRINT *, 'N:'
      PRINT *, N
      PRINT *, 'Checksum:'
      PRINT *, S
      PRINT *, 'Corners:'
      PRINT *, C(1, 1)
      PRINT *, C(N, N)
      
      END
      
      SUBROUTINE GENRND(MAT, N, MAXN, SEED)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL MAT(1024, 1024)
      INTEGER SEED
      INTEGER I, J
      INTEGER VAL
      
      DO I = 1, N
          DO J = 1, N
              VAL = (I * 17 + J * 23 + SEED) * 31
              MAT(I, J) = FLOAT(MOD(ABS(VAL), 10))
          ENDDO
      ENDDO
      
      END
      
      SUBROUTINE PRNTMT(MAT, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL MAT(1024, 1024)
      INTEGER I, J
      
      DO I = 1, N
          DO J = 1, N
              PRINT *, MAT(I, J)
          ENDDO
      ENDDO
      
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
      
      SUBROUTINE STRASN(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(1024, 1024)
      REAL B(1024, 1024)
      REAL C(1024, 1024)
      
      IF (N .LT. 64) THEN
          CALL MATSTD(A, B, C, N, MAXN)
      ELSE
          CALL MATBLK(A, B, C, N, MAXN)
      ENDIF
      
      END
      
      SUBROUTINE MATSTD(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(1024, 1024)
      REAL B(1024, 1024)
      REAL C(1024, 1024)
      INTEGER I, J, K
      REAL S
      
      DO I = 1, N
          DO J = 1, N
              S = 0.0
              DO K = 1, N
                  S = S + A(I, K) * B(K, J)
              ENDDO
              C(I, J) = S
          ENDDO
      ENDDO
      
      END
      
      SUBROUTINE MATBLK(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(1024, 1024)
      REAL B(1024, 1024)
      REAL C(1024, 1024)
      INTEGER BS
      INTEGER II, JJ, KK
      INTEGER I, J, K
      INTEGER IEND, JEND, KEND
      REAL AIK
      
      BS = 64
      
      DO I = 1, N
          DO J = 1, N
              C(I, J) = 0.0
          ENDDO
      ENDDO
      
      DO II = 1, N, BS
          IEND = MIN(II + BS - 1, N)
          DO KK = 1, N, BS
              KEND = MIN(KK + BS - 1, N)
              DO JJ = 1, N, BS
                  JEND = MIN(JJ + BS - 1, N)
                  DO I = II, IEND
                      DO K = KK, KEND
                          AIK = A(I, K)
                          DO J = JJ, JEND
                              C(I, J) = C(I, J) +
     1                            AIK * B(K, J)
                          ENDDO
                      ENDDO
                  ENDDO
              ENDDO
          ENDDO
      ENDDO
      
      END
