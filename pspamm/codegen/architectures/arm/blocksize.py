
class Old:
    @classmethod
    def getBlocksize(cls, m , n, bk, v_size, prec):

        bm = m
        bn = n
        
        if cls.ARM_condition(bm, bn, bk, v_size):
            while cls.ARM_condition(bm, bn, bk+1, v_size):
                bk += 1
            return (bm, bn, bk)

        while not cls.ARM_condition(bm, bn, bk, v_size):
            bm, bn = cls.lowerToNextDiv(m, n, bm, bn, v_size)

        while cls.ARM_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

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
    def ARM_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm + bn*bk <= 32

class Max:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        bm = 2
        bn = 1
        maxval = 0

        for i in range(v_size, m+1, v_size):
            for j in range(1, n+1):
                if cls.ARM_condition(i, j, bk, v_size):
                    if i*j > maxval:
                        maxval = i*j
                        bm = i
                        bn = j

        while cls.ARM_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)


    @classmethod
    def ARM_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn+bk) * vm + bn*bk <= 32

class MaxK:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        bm = 2
        bn = 1
        maxval = 0

        elem128 = 16 // prec.size()

        for i in range(v_size, m+1, v_size):
            for j in range(1, n+1):
                if cls.ARM_condition(i, j, bk, v_size, elem128):
                    if i*j > maxval:
                        maxval = i*j
                        bm = i
                        bn = j

        while cls.ARM_condition(bm, bn, bk+1, v_size, elem128):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def ARM_condition(cls, bm, bn, bk, v_size, elem128):
        # ceiling division
        vm = -(bm // -v_size)
        vk = -(bk // -elem128)
        return (bn+vk) * vm + bn*vk <= 32

Default = Max
