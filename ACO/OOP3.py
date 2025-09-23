class Car:
    def __init__(self):
        self.name=input("Enter the car name:")
        self.year=int(input("Enter the year:"))
        self.price=int(input("Enter the price:"))
        self.fuel_eff=float(input("Enter the fue efficency"))
    def display(self):
        print(f"Car Name:{self.name}")
        print(f"Manufactuer Year:{self.year}")
        print(f"price:{self.price}")
        print(f"Fuel efficency {self.fuel_eff}")
    def depreciation(self,saleValue,lifespan):
        print("Depreciation over time:",(self.price-saleValue)/lifespan)
c1=Car()
c1.display()
c1.depreciation(100000,25)