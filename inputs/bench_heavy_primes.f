      PROGRAM HPRIMES
      IMPLICIT NONE
      INTEGER I, J, COUNT, LIM, ISSQRT
      LOGICAL ISPRIME
      COUNT = 0
      LIM = 200000
      DO I = 2, LIM
          ISPRIME = .TRUE.
          ISSQRT = INT(SQRT(FLOAT(I)))
          J = 2
          DO WHILE (J .LE. ISSQRT)
              IF (MOD(I, J) .EQ. 0) THEN
                  ISPRIME = .FALSE.
                  GOTO 10
              ENDIF
              J = J + 1
          ENDDO
   10     CONTINUE
          IF (ISPRIME) COUNT = COUNT + 1
      ENDDO
      PRINT *, COUNT
      END
