      PROGRAM ARI
      IMPLICIT NONE
      INTEGER X
      X = 5
      IF (X - 5) 10, 20, 30
    10 PRINT *, -1
      GOTO 40
    20 PRINT *, 0
      GOTO 40
    30 PRINT *, 1
    40 CONTINUE
      END

