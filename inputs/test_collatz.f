      PROGRAM COLTZ
      IMPLICIT NONE
      INTEGER N, STEPS
      N = 27
      STEPS = 0
      DO WHILE (N .NE. 1)
          IF (MOD(N, 2) .EQ. 0) THEN
              N = N / 2
          ELSE
              N = 3 * N + 1
          ENDIF
          STEPS = STEPS + 1
      ENDDO
      PRINT *, STEPS
      N = 7
      STEPS = 0
      DO WHILE (N .NE. 1)
          IF (MOD(N, 2) .EQ. 0) THEN
              N = N / 2
          ELSE
              N = 3 * N + 1
          ENDIF
          STEPS = STEPS + 1
      ENDDO
      PRINT *, STEPS
      END
