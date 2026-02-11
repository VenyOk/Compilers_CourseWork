      PROGRAM NWTSQR
      IMPLICIT NONE
      REAL X, PREV, TARGET, DIFF
      INTEGER ITER
      TARGET = 2.0
      X = 1.0
      ITER = 0
      DO WHILE (ITER .LT. 20)
          PREV = X
          X = (X + TARGET / X) / 2.0
          DIFF = X - PREV
          IF (DIFF .LT. 0.0) DIFF = -DIFF
          IF (DIFF .LT. 0.00001) GOTO 10
          ITER = ITER + 1
      ENDDO
   10 CONTINUE
      PRINT *, X
      PRINT *, ITER
      END
