from chapter5.mul_layer import MulLayer
from decimal import Decimal

# for decimal
apple = Decimal('100')
apple_num = Decimal('2')
tax = Decimal('1.1')

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

