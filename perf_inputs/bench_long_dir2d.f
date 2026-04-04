      PROGRAM LDIR2D
      IMPLICIT NONE
      INTEGER I, J, T
      REAL U(224,224), A(224,224), B(224,224)
      REAL C(224,224), D(224,224), Y0(224,224), S

      DO I = 1, 224
          DO J = 1, 224
              U(I,J) = 0.0
              A(I,J) = 0.15
              B(I,J) = 0.15
              C(I,J) = 0.15
              D(I,J) = 0.15
              Y0(I,J) = 0.01
          ENDDO
      ENDDO

      DO I = 1, 224
          U(I,1) = 1.0
          U(I,224) = 1.0
      ENDDO

      DO J = 1, 224
          U(1,J) = 1.0
          U(224,J) = 1.0
      ENDDO

      DO T = 1, 1200
          DO I = 2, 223
              DO J = 2, 223
                  S = A(I,J)*U(I-1,J) + B(I,J)*U(I+1,J)
                  S = S + C(I,J)*U(I,J-1) + D(I,J)*U(I,J+1)
                  U(I,J) = S + Y0(I,J)
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(112,112)
      END
