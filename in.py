#a = input("Give me a number:\n")
a = input()

def addone(x):
    return x+1

with  open("technicaltest_ai_dxc/models","r") as file:
    lines = file.readlines()
    for i,line in enumerate(lines):
        print(f"({chr(i+ord("a"))})\t{line}")
        
        
    
        
        

print(addone(int(a)))
#a = input("Give me a number:\n")