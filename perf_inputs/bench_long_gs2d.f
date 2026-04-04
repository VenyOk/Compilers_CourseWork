      PROGRAM LGS2D
      IMPLICIT NONE
      INTEGER I, J, T
      REAL U(384,384)

      DO I = 1, 384
          DO J = 1, 384
              U(I,J) = 0.0
          ENDDO
      ENDDO

      DO I = 1, 384
          U(I,1) = 1.0
          U(I,384) = 1.0
      ENDDO

      DO J = 1, 384
          U(1,J) = 1.0
          U(384,J) = 1.0
      ENDDO

      DO T = 1, 3600
          DO I = 2, 383
              DO J = 2, 383
                  U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(192,192)
      END
