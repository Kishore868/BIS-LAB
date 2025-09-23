class Student:
    
    def __init__(self):
        self.name=input("Enter the name:")
        self.roll_no=int(input("Enter the roll number:"))
        self.m1=int(input("Enter the maths marks:"))
        self.m2=int(input("Enter the Physics marks:"))
        self.m3=int(input("Enter the chemistry marks:"))

    def Student_details(self):
        print(f"Name:{self.name}")
        print(f"Roll-Number:{self.roll_no}")
        print(f"Maths marks:{self.m1}")
        print(f"Physics marks:{self.m2}")
        print(f"Chemistry marks:{self.m3}")
    def total_marks(self):
        sum=self.m1+self.m2+self.m3
        print(f"The total marks of 3 subjects is:{sum}")
s1=Student()
s1.Student_details()
s1.total_marks()