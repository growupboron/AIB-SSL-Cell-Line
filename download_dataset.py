import wilds
from wilds import get_dataset

d = get_dataset("rxrx1", download=True)

print(d[0])
