      PROGRAM STRASS
      IMPLICIT NONE
      INTEGER MAXN
      PARAMETER (MAXN = 16)
      INTEGER N
      REAL A(16, 16), B(16, 16), C(16, 16)
      INTEGER I, J, SEED
      
C     Размер матрицы (степень двойки: 2, 4, 8, 16, 32)
      N = 8
      
C     Генерация случайных матриц
      SEED = 12345
      CALL GENRND(A, N, MAXN, SEED)
      SEED = SEED + 1000
      CALL GENRND(B, N, MAXN, SEED)
      
C     Вывод исходных матриц
      PRINT *, 'Matrix A:'
      CALL PRNTMT(A, N, MAXN)
      PRINT *, 'Matrix B:'
      CALL PRNTMT(B, N, MAXN)
      
C     Умножение матриц с помощью алгоритма Штрассена
      CALL STRASN(A, B, C, N, MAXN)
      
C     Вывод результата
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
      
C     Генератор маленьких чисел от 0 до 9
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
      
C     Базовый случай: матрицы 2x2
      IF (N .EQ. 2) THEN
          CALL STRAS2(A, B, C)
      ELSE
C     Разбиваем матрицы на подматрицы
      HALF = N / 2
      
C     Извлекаем подматрицы A
          DO I = 1, HALF
              DO J = 1, HALF
                  A11(I, J) = A(I, J)
                  A12(I, J) = A(I, J + HALF)
                  A21(I, J) = A(I + HALF, J)
                  A22(I, J) = A(I + HALF, J + HALF)
              ENDDO
          ENDDO
          
C     Извлекаем подматрицы B
          DO I = 1, HALF
              DO J = 1, HALF
                  B11(I, J) = B(I, J)
                  B12(I, J) = B(I, J + HALF)
                  B21(I, J) = B(I + HALF, J)
                  B22(I, J) = B(I + HALF, J + HALF)
              ENDDO
          ENDDO
          
C     Вычисляем промежуточные матрицы M1-M7 по алгоритму Штрассена
C     M1 = (A11 + A22) * (B11 + B22)
          CALL MADD(A11, A22, TMP1, HALF, MAXN)
          CALL MADD(B11, B22, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M1, HALF, MAXN)
          
C     M2 = (A21 + A22) * B11
          CALL MADD(A21, A22, TMP1, HALF, MAXN)
          CALL STRASN(TMP1, B11, M2, HALF, MAXN)
          
C     M3 = A11 * (B12 - B22)
          CALL MSUB(B12, B22, TMP1, HALF, MAXN)
          CALL STRASN(A11, TMP1, M3, HALF, MAXN)
          
C     M4 = A22 * (B21 - B11)
          CALL MSUB(B21, B11, TMP1, HALF, MAXN)
          CALL STRASN(A22, TMP1, M4, HALF, MAXN)
          
C     M5 = (A11 + A12) * B22
          CALL MADD(A11, A12, TMP1, HALF, MAXN)
          CALL STRASN(TMP1, B22, M5, HALF, MAXN)
          
C     M6 = (A21 - A11) * (B11 + B12)
          CALL MSUB(A21, A11, TMP1, HALF, MAXN)
          CALL MADD(B11, B12, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M6, HALF, MAXN)
          
C     M7 = (A12 - A22) * (B21 + B22)
          CALL MSUB(A12, A22, TMP1, HALF, MAXN)
          CALL MADD(B21, B22, TMP2, HALF, MAXN)
          CALL STRASN(TMP1, TMP2, M7, HALF, MAXN)
          
C     Вычисляем результирующие подматрицы
C     C11 = M1 + M4 - M5 + M7
          CALL MADD(M1, M4, TMP1, HALF, MAXN)
          CALL MSUB(TMP1, M5, TMP2, HALF, MAXN)
          CALL MADD(TMP2, M7, C11, HALF, MAXN)
          
C     C12 = M3 + M5
          CALL MADD(M3, M5, C12, HALF, MAXN)
          
C     C21 = M2 + M4
          CALL MADD(M2, M4, C21, HALF, MAXN)
          
C     C22 = M1 - M2 + M3 + M6
          CALL MSUB(M1, M2, TMP1, HALF, MAXN)
          CALL MADD(TMP1, M3, TMP2, HALF, MAXN)
          CALL MADD(TMP2, M6, C22, HALF, MAXN)
          
C     Объединяем подматрицы в результирующую матрицу C
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
      
C     Алгоритм Штрассена для матриц 2x2
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
