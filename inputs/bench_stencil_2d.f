C     2D Stencil - Gauss-Seidel for Laplace equation
C     Tests: loop interchange, tiling, skewing
      PROGRAM STNCL2
      INTEGER I, J, K, N, NITER
      REAL U(64, 64), V(64, 64), DIFF, MDIFF

C     Initialize boundary and interior
      N = 64
      NITER = 5
      DO I = 1, 64
          DO J = 1, 64
              U(I, J) = 0.0
              V(I, J) = 0.0
          ENDDO
      ENDDO
      DO J = 1, 64
          U(1, J) = 1.0
          U(64, J) = 1.0
          V(1, J) = 1.0
          V(64, J) = 1.0
      ENDDO
      DO I = 1, 64
          U(I, 1) = 1.0
          U(I, 64) = 1.0
          V(I, 1) = 1.0
          V(I, 64) = 1.0
      ENDDO

C     Gauss-Seidel iterations
      DO K = 1, NITER
          DO I = 2, 63
              DO J = 2, 63
                  V(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              ENDDO
          ENDDO
C         Copy back
          DO I = 2, 63
              DO J = 2, 63
                  U(I, J) = V(I, J)
              ENDDO
          ENDDO
      ENDDO

C     Compute max residual
      MDIFF = 0.0
      DO I = 2, 63
          DO J = 2, 63
              DIFF = ABS(U(I, J) - V(I, J))
              IF (DIFF .GT. MDIFF) THEN
                  MDIFF = DIFF
              ENDIF
          ENDDO
      ENDDO

      PRINT *, 'Stencil done'
      PRINT *, 'Center:', U(32, 32)
      PRINT *, 'Residual:', MDIFF

      END
