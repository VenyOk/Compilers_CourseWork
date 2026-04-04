C     Gauss-Seidel iterative solver for Laplace equation
C     Classic iterative nest: outer loop K is the iteration counter,
C     inner loops I,J update U(I,J) using neighbors - backward dep on I-1,J-1
      PROGRAM GSEIDEL
      INTEGER I, J, K, N
      REAL U(128, 128), UOLD(128, 128), ERR, MAXERR

      N = 128

      DO I = 1, 128
          DO J = 1, 128
              U(I, J) = 0.0
              UOLD(I, J) = 0.0
          ENDDO
      ENDDO
      DO J = 1, 128
          U(1, J) = 1.0
          U(128, J) = 1.0
      ENDDO
      DO I = 1, 128
          U(I, 1) = 1.0
          U(I, 128) = 1.0
      ENDDO

C     50 Gauss-Seidel iterations
      DO K = 1, 50
          DO I = 2, 127
              DO J = 2, 127
                  U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              ENDDO
          ENDDO
      ENDDO

C     Check convergence
      MAXERR = 0.0
      DO I = 2, 127
          DO J = 2, 127
              ERR = ABS(U(I,J) - UOLD(I,J))
              IF (ERR .GT. MAXERR) THEN
                  MAXERR = ERR
              ENDIF
          ENDDO
      ENDDO

      PRINT *, 'Gauss-Seidel done, center value:', U(64, 64)
      PRINT *, 'Max error:', MAXERR
      END
