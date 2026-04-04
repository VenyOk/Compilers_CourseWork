      PROGRAM LGS3D
      IMPLICIT NONE
      INTEGER I, J, K, T
      REAL U(96,96,96), S

      DO I = 1, 96
          DO J = 1, 96
              DO K = 1, 96
                  U(I,J,K) = 0.0
              ENDDO
          ENDDO
      ENDDO

      DO I = 1, 96
          DO J = 1, 96
              U(I,J,1) = 1.0
              U(I,J,96) = 1.0
          ENDDO
      ENDDO

      DO I = 1, 96
          DO K = 1, 96
              U(I,1,K) = 1.0
              U(I,96,K) = 1.0
          ENDDO
      ENDDO

      DO J = 1, 96
          DO K = 1, 96
              U(1,J,K) = 1.0
              U(96,J,K) = 1.0
          ENDDO
      ENDDO

      DO T = 1, 200
          DO I = 2, 95
              DO J = 2, 95
                  DO K = 2, 95
                      S = U(I-1,J,K) + U(I+1,J,K) + U(I,J-1,K)
                      S = S + U(I,J+1,K) + U(I,J,K-1) + U(I,J,K+1)
                      U(I,J,K) = S / 6.0
                  ENDDO
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(48,48,48)
      END
