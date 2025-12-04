      PROGRAM PRIME
      IMPLICIT NONE
      INTEGER N, I, J, ISPRIM, COUNT
      N = 20
      COUNT = 0
      DO I = 2, N
          ISPRIM = 1
          IF (I .GT. 2) THEN
              DO J = 2, I - 1
                  IF ((I / J) * J .EQ. I) THEN
                      ISPRIM = 0
                  ENDIF
              ENDDO
          ENDIF
          IF (ISPRIM .EQ. 1) THEN
              COUNT = COUNT + 1
              PRINT *, I
          ENDIF
      ENDDO
      PRINT *, COUNT
      END

