      PROGRAM NUMCLS
      IMPLICIT NONE
      INTEGER I, J, EVENS, ODDS, DIV3, PRIMES
      INTEGER ISPRIM
      LOGICAL PRIME
      EVENS = 0
      ODDS = 0
      DIV3 = 0
      PRIMES = 0
      DO I = 1, 20
          IF (MOD(I, 2) .EQ. 0) THEN
              EVENS = EVENS + 1
          ELSE
              ODDS = ODDS + 1
          ENDIF
          IF (MOD(I, 3) .EQ. 0) THEN
              DIV3 = DIV3 + 1
          ENDIF
          IF (I .GE. 2) THEN
              PRIME = .TRUE.
              DO J = 2, I - 1
                  IF (MOD(I, J) .EQ. 0) THEN
                      PRIME = .FALSE.
                  ENDIF
              ENDDO
              IF (PRIME) THEN
                  PRIMES = PRIMES + 1
              ENDIF
          ENDIF
      ENDDO
      PRINT *, EVENS
      PRINT *, ODDS
      PRINT *, DIV3
      PRINT *, PRIMES
      END
