class Max:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        # v_size default is 2, however for SVE that parameter will always be larger
        bm = 2
        bn = 1
        maxval = 0

        for i in range(1, m + 1, 1):
            next_multiple = i
            while next_multiple % v_size != 0:
                next_multiple += 1
            for j in range(1, n + 1):
                if cls.ARM_condition(next_multiple, j, bk, v_size) and cls.tileable(m, i):
                    if i * j >= maxval:
                        maxval = i * j
                        bm = i
                        bn = j

        if maxval == 0:
            raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")

        while cls.ARM_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def ARM_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)  
        return (bn + bk) * vm + bn*bk <= 32

    @classmethod
    def tileable(cls, m, bm):
        return m % bm == 0

class MaxK:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        # v_size default is 2, however for SVE that parameter will always be larger
        bm = 2
        bn = 1
        maxval = 0

        elem128 = 16 // prec.size()

        for i in range(1, m + 1, 1):
            next_multiple = -(i // -v_size) * v_size
            for j in range(1, n + 1):
                if cls.ARM_condition(next_multiple, j, bk, v_size, elem128) and cls.tileable(m, i):
                    if i * j >= maxval:
                        maxval = i * j
                        bm = i
                        bn = j

        if maxval == 0:
            raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")

        while cls.ARM_condition(bm, bn, bk+1, v_size, elem128):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def ARM_condition(cls, bm, bn, bk, v_size, elem128):
        # ceiling division
        vkext = -(bk // -elem128)
        isvkext = bn*vkext <= 16 if elem128 == 2 else bn*vkext <= 8
        vm = -(bm // -v_size)
        vk = vkext if isvkext else bk
        return (bn + bk) * vm + bn*vk <= 32

    @classmethod
    def tileable(cls, m, bm):
        return m % bm == 0

class Cube:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        # v_size default is 2, however for SVE that parameter will always be larger
        bm = 2
        bn = 1
        maxval = 0

        elem128 = 16 // prec.size()

        for i in range(1, m + 1, 1):
            next_multiple = -(i // -v_size) * v_size
            for j in range(1, n + 1):
                for k in range(1, 200):
                    if cls.ARM_condition(next_multiple, j, k, v_size, elem128) and cls.tileable(m, i):
                        if i * j * k >= maxval:
                            maxval = i * j * k
                            bm = i
                            bn = j
                            bk = k

        if maxval == 0:
            raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")

        return (bm, bn, bk)

    @classmethod
    def ARM_condition(cls, bm, bn, bk, v_size, elem128):
        # ceiling division
        vkext = -(bk // -elem128)
        isvkext = bn*vkext <= 16 if elem128 == 2 else bn*vkext <= 8
        vm = -(bm // -v_size)
        vk = vkext if isvkext else bk
        return (bn + bk) * vm + bn*vk <= 32

    @classmethod
    def tileable(cls, m, bm):
        return m % bm == 0

Default = MaxK
