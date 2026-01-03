import random
count = 1000
sum = 0
for j in range(10):
    for i in range(count):
        p = random.randint(0, 100)
        sum += p
    sum = sum / count
    print(sum)

