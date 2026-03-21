      PROGRAM MTD2D
      IMPLICIT NONE
      INTEGER I, J, T
      REAL U(160,160), A(160,160), B(160,160)
      REAL C(160,160), D(160,160), Y0(160,160), S

      DO I = 1, 160
          DO J = 1, 160
              U(I,J) = 0.0
              A(I,J) = 0.15
              B(I,J) = 0.15
              C(I,J) = 0.15
              D(I,J) = 0.15
              Y0(I,J) = 0.01
          ENDDO
      ENDDO

      DO I = 1, 160
          U(I,1) = 1.0
          U(I,160) = 1.0
      ENDDO

      DO J = 1, 160
          U(1,J) = 1.0
          U(160,J) = 1.0
      ENDDO

      DO T = 1, 30
          DO I = 2, 159
              DO J = 2, 159
                  S = A(I,J)*U(I-1,J) + B(I,J)*U(I+1,J)
                  S = S + C(I,J)*U(I,J-1) + D(I,J)*U(I,J+1)
                  U(I,J) = S + Y0(I,J)
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(80,80)
      END
