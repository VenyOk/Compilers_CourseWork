      PROGRAM GCDT
      IMPLICIT NONE
      INTEGER A, B, R
      A = 48
      B = 18
      CALL EUCGCD(A, B, R)
      PRINT *, R
      A = 100
      B = 75
      CALL EUCGCD(A, B, R)
      PRINT *, R
      A = 17
      B = 13
      CALL EUCGCD(A, B, R)
      PRINT *, R
      END

      SUBROUTINE EUCGCD(A, B, R)
      IMPLICIT NONE
      INTEGER A, B, R, TMP
      R = A
      TMP = B
      DO WHILE (TMP .NE. 0)
          A = R
          R = TMP
          TMP = MOD(A, TMP)
      ENDDO
      END
