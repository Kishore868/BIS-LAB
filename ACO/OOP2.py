class Bankaccount:
    def __init__(self,name,balance):
        self.name=name
        self.balance=balance
    def deposit(self):
        deposit_amount=int(input("Enter the deposit amount:"))
        self.balance+=deposit_amount
        print(f"The current balance is:{self.balance}")
    def withdraw_money(self):
        withdraw_amount=int(input("Enter the withdraw amount:"))
        if(withdraw_amount<=self.balance):
             self.balance-=withdraw_amount
        else:
            print("The required amount not avaiable in your account")
        print(f"The current balance is:{self.balance}")
    def Checkbalance(self):
        print(f"The balance in your account is{self.balance}")

p1=Bankaccount("karthik",200000)
p1.deposit()
p1.withdraw_money()
p1.Checkbalance()
