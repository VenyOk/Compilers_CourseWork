      PROGRAM FCONST
      IMPLICIT NONE
      INTEGER X
      X = DOUBLE(5)
      PRINT *, X
      X = TRIPLE(4)
      PRINT *, X
      X = DOUBLE(7) + TRIPLE(3)
      PRINT *, X
      END

      INTEGER FUNCTION DOUBLE(N)
      IMPLICIT NONE
      INTEGER N
      DOUBLE = N * 2
      RETURN
      END

      INTEGER FUNCTION TRIPLE(N)
      IMPLICIT NONE
      INTEGER N
      TRIPLE = N * 3
      RETURN
      END
