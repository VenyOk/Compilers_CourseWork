      PROGRAM ARROPS
      IMPLICIT NONE
      INTEGER A(10), B(10), C(10)
      INTEGER I, MAXVAL, MINVAL
      DO I = 1, 10
          A(I) = I * 2
          B(I) = I + 5
          C(I) = A(I) + B(I)
      ENDDO
      MAXVAL = A(1)
      MINVAL = A(1)
      DO I = 2, 10
          IF (A(I) .GT. MAXVAL) THEN
              MAXVAL = A(I)
          ENDIF
          IF (A(I) .LT. MINVAL) THEN
              MINVAL = A(I)
          ENDIF
      ENDDO
      PRINT *, MAXVAL
      PRINT *, MINVAL
      PRINT *, C(5)
      END

