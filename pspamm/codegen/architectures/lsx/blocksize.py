class Max:
    @classmethod
    def getBlocksize(cls, m, n, bk, v_size, prec):
        bm = v_size
        bn = 1
        maxval = 0

        for i in range(v_size, m+1, v_size):
            for j in range(1, n+1):
                # can be replaced by cls.LSX_condition_extended here
                # (but that seemed to be slower in the end)
                if cls.LSX_condition(i, j, bk, v_size):
                    if i*j > maxval and (cls.LSX_condition(i, j, bk, v_size) or j > 1):
                        maxval = i*j
                        bm = i
                        bn = j 

        while cls.LSX_condition(bm, bn, bk+1, v_size):
            bk += 1

        return (bm, bn, bk)

    @classmethod
    def LSX_condition(cls, bm, bn, bk, v_size):
        # ceiling division
        vm = -(bm // -v_size)
        return (bn + bk) * vm + bn * bk <= 32

Default = Max
