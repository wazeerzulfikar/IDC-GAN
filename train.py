from model import *

Generator = create_generator(1,1,4)
print(g.summary())
print("-------------------------")
Discriminator = create_discriminator(1,1,4,2)
print(d.summary())

