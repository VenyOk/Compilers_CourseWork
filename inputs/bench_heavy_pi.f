      PROGRAM HPI
      IMPLICIT NONE
      INTEGER I
      REAL PI, TERM
      PI = 0.0
      DO I = 1, 5000000
          TERM = 1.0 / (2.0 * FLOAT(I) - 1.0)
          IF (MOD(I, 2) .EQ. 1) THEN
              PI = PI + TERM
          ELSE
              PI = PI - TERM
          ENDIF
      ENDDO
      PI = PI * 4.0
      PRINT *, PI
      END
