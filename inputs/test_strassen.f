      PROGRAM STRASS
      IMPLICIT NONE
      INTEGER MAXN
      PARAMETER (MAXN = 16)
      INTEGER N
      REAL A(16, 16), B(16, 16), C(16, 16)
      INTEGER I, J, SEED
      
      N = 8
      
      SEED = 12345
      CALL GENRND(A, N, MAXN, SEED)
      SEED = SEED + 1000
      CALL GENRND(B, N, MAXN, SEED)
      
      PRINT *, 'Matrix A:'
      CALL PRNTMT(A, N, MAXN)
      PRINT *, 'Matrix B:'
      CALL PRNTMT(B, N, MAXN)
      
      CALL STRASN(A, B, C, N, MAXN)
      
      PRINT *, 'Result C = A * B:'
      CALL PRNTMT(C, N, MAXN)
      
      END
      
      SUBROUTINE GENRND(MAT, N, MAXN, SEED)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL MAT(16, 16)
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
      REAL MAT(16, 16)
      INTEGER I, J
      
      DO I = 1, N
          DO J = 1, N
              PRINT *, MAT(I, J)
          ENDDO
      ENDDO
      
      END
      
      SUBROUTINE STRASN(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(16, 16), B(16, 16), C(16, 16)
      INTEGER HALF
      REAL A11(16, 16), A12(16, 16), A21(16, 16), A22(16, 16)
      REAL B11(16, 16), B12(16, 16), B21(16, 16), B22(16, 16)
      REAL C11(16, 16), C12(16, 16), C21(16, 16), C22(16, 16)
      REAL M1(16, 16), M2(16, 16), M3(16, 16), M4(16, 16)
      REAL M5(16, 16), M6(16, 16), M7(16, 16)
      REAL TMP1(16, 16), TMP2(16, 16)
      INTEGER I, J
      
      IF (N .EQ. 2) THEN
          CALL STRAS2(A, B, C)
      ELSE
      HALF = N / 2
      
          DO I = 1, HALF
              DO J = 1, HALF
                  A11(I, J) = A(I, J)
                  A12(I, J) = A(I, J + HALF)
                  A21(I, J) = A(I + HALF, J)
                  A22(I, J) = A(I + HALF, J + HALF)
              ENDDO
          ENDDO
          
          DO I = 1, HALF
              DO J = 1, HALF
                  B11(I, J) = B(I, J)
                  B12(I, J) = B(I, J + HALF)
                  B21(I, J) = B(I + HALF, J)
                  B22(I, J) = B(I + HALF, J + HALF)
              ENDDO
          ENDDO
          
          CALL MADD(A11, A22, TMP1, HALF, MAXN)
          CALL MADD(B11, B22, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M1, HALF, MAXN)
          
          CALL MADD(A21, A22, TMP1, HALF, MAXN)
          CALL STRASN(TMP1, B11, M2, HALF, MAXN)
          
          CALL MSUB(B12, B22, TMP1, HALF, MAXN)
          CALL STRASN(A11, TMP1, M3, HALF, MAXN)
          
          CALL MSUB(B21, B11, TMP1, HALF, MAXN)
          CALL STRASN(A22, TMP1, M4, HALF, MAXN)
          
          CALL MADD(A11, A12, TMP1, HALF, MAXN)
          CALL STRASN(TMP1, B22, M5, HALF, MAXN)
          
          CALL MSUB(A21, A11, TMP1, HALF, MAXN)
          CALL MADD(B11, B12, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M6, HALF, MAXN)
          
          CALL MSUB(A12, A22, TMP1, HALF, MAXN)
          CALL MADD(B21, B22, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M7, HALF, MAXN)
          
          CALL MADD(M1, M4, TMP1, HALF, MAXN)
          CALL MSUB(TMP1, M5, TMP2, HALF, MAXN)
          CALL MADD(TMP2, M7, C11, HALF, MAXN)
          
          CALL MADD(M3, M5, C12, HALF, MAXN)
          
          CALL MADD(M2, M4, C21, HALF, MAXN)
          
          CALL MSUB(M1, M2, TMP1, HALF, MAXN)
          CALL MADD(TMP1, M3, TMP2, HALF, MAXN)
          CALL MADD(TMP2, M6, C22, HALF, MAXN)
          
          DO I = 1, HALF
              DO J = 1, HALF
                  C(I, J) = C11(I, J)
                  C(I, J + HALF) = C12(I, J)
                  C(I + HALF, J) = C21(I, J)
                  C(I + HALF, J + HALF) = C22(I, J)
              ENDDO
          ENDDO
      ENDIF
      
      END
      
      SUBROUTINE STRAS2(A, B, C)
      IMPLICIT NONE
      REAL A(16, 16), B(16, 16), C(16, 16)
      REAL M1, M2, M3, M4, M5, M6, M7
      
      M1 = (A(1, 1) + A(2, 2)) * (B(1, 1) + B(2, 2))
      M2 = (A(2, 1) + A(2, 2)) * B(1, 1)
      M3 = A(1, 1) * (B(1, 2) - B(2, 2))
      M4 = A(2, 2) * (B(2, 1) - B(1, 1))
      M5 = (A(1, 1) + A(1, 2)) * B(2, 2)
      M6 = (A(2, 1) - A(1, 1)) * (B(1, 1) + B(1, 2))
      M7 = (A(1, 2) - A(2, 2)) * (B(2, 1) + B(2, 2))
      
      C(1, 1) = M1 + M4 - M5 + M7
      C(1, 2) = M3 + M5
      C(2, 1) = M2 + M4
      C(2, 2) = M1 - M2 + M3 + M6
      
      END
      
      SUBROUTINE MADD(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(16, 16), B(16, 16), C(16, 16)
      INTEGER I, J
      
      DO I = 1, N
          DO J = 1, N
              C(I, J) = A(I, J) + B(I, J)
          ENDDO
      ENDDO
      
      END
      
      SUBROUTINE MSUB(A, B, C, N, MAXN)
      IMPLICIT NONE
      INTEGER N, MAXN
      REAL A(16, 16), B(16, 16), C(16, 16)
      INTEGER I, J
      
      DO I = 1, N
          DO J = 1, N
              C(I, J) = A(I, J) - B(I, J)
          ENDDO
      ENDDO
      
      END
