      PROGRAM MTL2D
      IMPLICIT NONE
      INTEGER I, J, T
      REAL U(192,192)

      DO I = 1, 192
          DO J = 1, 192
              U(I,J) = 0.0
          ENDDO
      ENDDO

      DO I = 1, 192
          U(I,1) = 1.0
          U(I,192) = 1.0
      ENDDO

      DO J = 1, 192
          U(1,J) = 1.0
          U(192,J) = 1.0
      ENDDO

      DO T = 1, 40
          DO I = 2, 191
              DO J = 2, 191
                  U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(96,96)
      END
