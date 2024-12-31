class MaxBn:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = v_size
        bn = 1

        for j in range(1, n+1):
            if cls.RVV_condition(bm, j, bk, v_size):
                bn = j

        while cls.RVV_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def RVV_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm <= 32 and bn*bk + 2 <= 32

class CubeBn:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = v_size
        bn = 1

        maxval = 0

        for j in range(1, n+1):
            for k in range(1, 200):
                if cls.RVV_condition(bm, j, k, v_size):
                    if j*k >= maxval:
                        maxval = j*k
                        bn = j
                        bk = k

        return (bm, bn, bk)

    @classmethod
    def RVV_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm <= 32 and bn*bk + 2 <= 32

Default = MaxBn
