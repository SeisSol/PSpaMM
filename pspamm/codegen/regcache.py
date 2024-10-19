
class RegisterCache:
    def __init__(self, registers):
        self.access = 0
        self.lru = [-1] * len(registers)
        self.registers = registers
        self.storage = {}
    
    def get(self, value):
        self.access += 1

        evicted = False

        if value not in self.storage:
            evicted = True
            minaccess = self.access
            minidx = -1
            for i, last in enumerate(self.lru):
                if last < minaccess:
                    minaccess = last
                    minidx = i
            self.storage[value] = minidx

        regidx = self.storage[value]

        self.lru[regidx] = self.access

        return (self.registers[regidx], evicted)
