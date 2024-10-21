class Old:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = m
        bn = n
        
        if cls.KNL_condition(bm, bn, bk, v_size):
            while cls.KNL_condition(bm, bn, bk+1, v_size):
                bk += 1
            return (bm, bn)

        while not cls.KNL_condition(bm, bn, bk, v_size):
            bm, bn = cls.lowerToNextDiv(m, n, bm, bn, v_size)

        while cls.KNL_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn)

    @classmethod
    def lowerToNextDiv(cls, m, n, bm, bn, v_size):
        if bm > bn and bm > v_size:
            bm -= v_size
            while m % bm != 0:
                bm -= v_size
        else:
            bn -= 1
            while n % bn != 0:
                bn -= 1

        return bm, bn

    @classmethod
    def KNL_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm  <= 32

class Max:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = 8
        bn = 1
        maxval = 0

        for i in range(1, m+1):
            next_multiple = -(bm // -v_size)
            for j in range(1, n+1):
                if cls.KNL_condition(next_multiple, j, bk, v_size) and cls.tileable(m, bm):
                    if i*j > maxval:
                        maxval = i*j
                        bm = i
                        bn = j 
        
        while cls.KNL_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def KNL_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm <= 32

    @classmethod
    def tileable(cls, m, bm):
        return m % bm == 0

class MaxBn:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = v_size
        bn = 1

        for j in range(1, n+1):
            if cls.KNL_condition(bm, j, bk, v_size):
                bn = j

        while cls.KNL_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def KNL_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm <= 32

class CubeBn:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = v_size
        bn = 1

        maxval = 0

        for j in range(1, n+1):
            for k in range(1, 200):
                if cls.KNL_condition(bm, j, k, v_size):
                    if j*k > maxval:
                        maxval = j*k
                        bn = j
                        bk = k

        return (bm, bn, bk)

    @classmethod
    def KNL_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm <= 32

Default = MaxBn
