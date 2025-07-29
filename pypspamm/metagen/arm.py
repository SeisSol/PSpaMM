
def arm_basic():
    generator = MetaGenerator()

    generator.add_condition('', 'arm128')
    generator.add_condition('svcntb() == 16', 'arm_sve128')
    generator.add_condition('svcntb() == 32', 'arm_sve256')
    generator.add_condition('svcntb() == 64', 'arm_sve512')
    generator.add_condition('svcntb() == 128', 'arm_sve1024')
    generator.add_condition('svcntb() == 256', 'arm_sve2048')
