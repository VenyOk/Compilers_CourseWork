      PROGRAM STRASS
      IMPLICIT NONE
      INTEGER A(2, 2), B(2, 2), C(2, 2)
      INTEGER I, J
      
      ! Простые тестовые матрицы 2x2
      ! A = [1  2]
      !     [3  4]
      A(1, 1) = 1
      A(1, 2) = 2
      A(2, 1) = 3
      A(2, 2) = 4
      
      ! B = [5  6]
      !     [7  8]
      B(1, 1) = 5
      B(1, 2) = 6
      B(2, 1) = 7
      B(2, 2) = 8
      
      ! Ожидаемый результат C = A * B:
      ! C[1,1] = 1*5 + 2*7 = 5 + 14 = 19
      ! C[1,2] = 1*6 + 2*8 = 6 + 16 = 22
      ! C[2,1] = 3*5 + 4*7 = 15 + 28 = 43
      ! C[2,2] = 3*6 + 4*8 = 18 + 32 = 50
      ! C = [19  22]
      !     [43  50]
      
      PRINT *, 'Matrix A:'
      DO I = 1, 2
          DO J = 1, 2
              PRINT *, A(I, J)
          ENDDO
      ENDDO
      
      PRINT *, 'Matrix B:'
      DO I = 1, 2
          DO J = 1, 2
              PRINT *, B(I, J)
          ENDDO
      ENDDO
      
      ! Умножение матриц с помощью алгоритма Штрассена
      CALL STRAS2(A, B, C)
      
      PRINT *, 'Result C = A * B (expected: 19, 22, 43, 50):'
      DO I = 1, 2
          DO J = 1, 2
              PRINT *, C(I, J)
          ENDDO
      ENDDO
      
      END
      
      SUBROUTINE STRAS2(A, B, C)
      IMPLICIT NONE
      INTEGER A(2, 2), B(2, 2), C(2, 2)
      INTEGER M1, M2, M3, M4, M5, M6, M7
      
      ! Алгоритм Штрассена для матриц 2x2
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

