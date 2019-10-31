a = ['1111_mask.png', '2222_mask.png']
b = []
for i in a:
    b.append(i.replace('_mask.png', '.png'))
print(b)