      PROGRAM GS3D
      IMPLICIT NONE
      INTEGER I, J, K, T
      REAL U(40,40,40), S

      DO I = 1, 40
          DO J = 1, 40
              DO K = 1, 40
                  U(I,J,K) = 0.0
              ENDDO
          ENDDO
      ENDDO

      DO I = 1, 40
          DO J = 1, 40
              U(I,J,1) = 1.0
              U(I,J,40) = 1.0
          ENDDO
      ENDDO

      DO I = 1, 40
          DO K = 1, 40
              U(I,1,K) = 1.0
              U(I,40,K) = 1.0
          ENDDO
      ENDDO

      DO J = 1, 40
          DO K = 1, 40
              U(1,J,K) = 1.0
              U(40,J,K) = 1.0
          ENDDO
      ENDDO

      DO T = 1, 8
          DO I = 2, 39
              DO J = 2, 39
                  DO K = 2, 39
                      S = U(I-1,J,K) + U(I+1,J,K) + U(I,J-1,K)
                      S = S + U(I,J+1,K) + U(I,J,K-1) + U(I,J,K+1)
                      U(I,J,K) = S / 6.0
                  ENDDO
              ENDDO
          ENDDO
      ENDDO

      WRITE(*,*) U(20,20,20)
      END
