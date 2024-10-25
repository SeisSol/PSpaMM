class Old:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):

        bm = m
        bn = n
        
        if cls.HSW_condition(bm, bn, bk, v_size):
            while cls.HSW_condition(bm, bn, bk+1, v_size):
                bk += 1
            return (bm, bn)

        while not cls.HSW_condition(bm, bn, bk, v_size):
            bm, bn = cls.lowerToNextDiv(m, n, bm, bn, v_size)

        while cls.HSW_condition(bm, bn, bk+1, v_size):
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
    def HSW_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn + bk) * vm + bn * bk <= 16

class Max:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        bm = 4
        bn = 1
        maxval = 0

        for i in range(v_size, m+1, v_size):
            for j in range(1, n+1):
                # can be replaced by cls.HSW_condition_extended here
                # (but that seemed to be slower in the end)
                if cls.HSW_condition(i, j, bk, v_size):
                    if i*j > maxval and (cls.HSW_condition(i, j, bk, v_size) or j > 1):
                        maxval = i*j
                        bm = i
                        bn = j 

        while cls.HSW_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def HSW_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn + bk) * vm + bn * bk <= 16

    @classmethod
    def HSW_condition_extended(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return bn * vm + bn * bk + 1 <= 16

class Cube:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        bm = 4
        bn = 1
        maxval = 0

        for i in range(v_size, m+1, v_size):
            for j in range(1, n+1):
                for k in range(1, 200):
                    # can be replaced by cls.HSW_condition_extended here
                    # (but that seemed to be slower in the end)
                    if cls.HSW_condition(i, j, bk, v_size):
                        if i*j*k >= maxval and (cls.HSW_condition(i, j, k, v_size) or j > 1):
                            maxval = i*j*k
                            bm = i
                            bn = j
                            bk = k

        return (bm, bn, bk)

    @classmethod
    def HSW_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn + bk) * vm + bn * bk <= 16

    @classmethod
    def HSW_condition_extended(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return bn * vm + bn * bk + 1 <= 16

Default = Max
