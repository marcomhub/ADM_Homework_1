#Say "Hello, World!" With Python

print ("Hello, World!")

#Python If-Else

#!/bin/python

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n % 2 != 0:
    print("Weird")
elif n % 2 == 0 and n in range(2,5) or n>20:
    print("Not Weird")
elif n % 2 == 0 and n in range(6,21):
    print("Weird")
    
#Arithmetic Operators

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python: Division

from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a//b)
    print(a/b)

#Loops

if __name__ == '__main__':
    n = int(raw_input())
    for i in range(0,n):
        print(i**2)
#Write a function

def is_leap(year):
    return year%400==0 or (year%4==0 and year%100!=0)
    
#Print Function

if __name__ == '__main__':
    n = int(input())
    vector=[]
    for i in range(1,n+1):
        vector.append(i)
    print(''.join(map(str,vector)))
    
#List Comprehension

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
print([[i,j,k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1) if i+j+k!=n])

#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
b=sorted(arr)
i=1
while b[-i] == max(b):
    i=i+1
print(b[-i])

#sWAP cASE

def swap_case(s):
    return s.swapcase()

#String Split and Join

def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line
  
#What's Your Name?

def print_full_name(a, b):
    print("Hello",a,b+"!",'You just delved into python.')

#Mutations

def mutate_string(string, position, character):
    string = string[:position]+character+string[(position+1):]
    return string

#Find a string

def count_substring(string, sub_string):
    count=0
    p=0
    while p < len(string):
        i=string.find(sub_string, p)
        if i>-1:
            count=count+1
            p=i+1
        else: break
    return count

#String Validators

if __name__ == '__main__':
    s = input()
    alnum=False
    al=False
    dig=False
    low=False
    up=False
for i in range(0,len(s)):
    if s[i].isalnum():
        alnum=True
    if s[i].isalpha():
        al=True
    if s[i].isdigit():
        dig=True
    if s[i].islower():
        low=True
    if s[i].isupper():
        up=True
print (alnum)
print (al)
print (dig)
print (low)
print (up)

#Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap

def wrap(string, max_width):
    i=max_width
    a=string[0:max_width]+'\n'
    while i<=len(string):
        a=a+(string[i:i+max_width]+'\n')
        i=i+max_width
    return a

#Designer Door Mat

N, M = map(int,input().split())
for i in range(1,N,2):
    print ((i * ".|.").center(M,"-"))
print ("WELCOME".center(M,"-"))
for i in range(N,1,-2):
    print (((i-2) * ".|.").center(M,"-"))
     
#Introduction to Sets

def average(array):
    return sum(set(arr))/len(set(arr))

#Symmetric Difference

M,m=(input(),input().split())
N,n=(input(),input().split())
set1=set(m)
set2=set(n)
mn=set1.difference(set2)
nm=set2.difference(set1)
ris=nm.union(mn)
print ("\n".join(sorted(ris,key=int)))

#Set .add() 

n=int(input())
s = set()
for _ in range(0,n):
    s.add(input())
print(len(s))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N= int(input())
for i in range(0,N):
    com=input().split()
    if com[0]=='remove':
        s.remove(int(com[1]))
    elif com[0]=='discard':
        s.discard(int(com[1]))
    else:
        s.pop()
print(sum(s))

#No Idea!

n,m=map(int,input().split())
arr=input().split()
A=set(input().split())
B=set(input().split())
happy=0
for i in range(0,len(arr)):
    if arr[i] in A:
        happy=happy+1
    elif arr[i] in B:
        happy=happy-1
print(happy)

#Set .difference() Operation

n=int(input())
N=set(input().split())
c=int(input())
C=set(input().split())
print(len(N.difference(C)))

#Set .symmetric_difference() Operation

n=int(input())
N=set(input().split())
c=int(input())
C=set(input().split())
print(len(N.symmetric_difference(C)))

#Set Mutations

a=int(input())
A=set(map(int,input().split()))
s=int(input())

for i in range(s):
    C,c=input().split()
    st=set(map(int,input().split()))
    if C=="intersection_update":
        A.intersection_update(st)
    elif C=="update":
        A.update(st)
    elif C=="symmetric_difference_update":
        A.symmetric_difference_update(st)
    elif C=="difference_update":
        A.difference_update(st)
print(sum(A))

#The Captain's Room 

k=int(input())
arr=input().split()
h=set(arr)
count={}
for r in arr:
   if r in count:
      count[r]+=1
   else:
      count[r]=1
for r in arr:
    if count[r]==1:
        print(r)

#Check Subset

n=int(input())
for i in range(n):
    a=int(input())
    A=set(input().split())
    b=int(input())
    B=set(input().split())
    if B.intersection(A)==A:
        r=True
    else:
        r=False
    print(r)

#Check Strict Superset

A=set(input().split())
n=int(input())
for i in range(n):
    B=set(input().split())
    if A.intersection(B)==B and len(A)>len(B):
        r=True
    else:
        r=False
        break
print(r)

#Exceptions

n=int(input())
for i in range(n):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

#Map and Lambda Function

cube = lambda x: x*x*x

def fibonacci(n):
    fib=[]
    if n==1:
        fib=[0]
    elif n==2:
        fib=[0,1]
    elif n>2:
        fib=[0,1]
        for i in range(2,n):
            fib.append((fib[i-1]+fib[i-2]))
        
    return fib

#Calendar Module

import calendar
MM,DD,YYYY=map(int,input().split())
d=calendar.weekday(YYYY,MM,DD)
if d==0:
    print('MONDAY')
if d==1:
    print('TUESDAY')
if d==2:
    print('WEDNESDAY')
if d==3:
    print('THURSDAY')
if d==4:
    print('FRIDAY')
if d==5:
    print('SATURDAY')
if d==6:
    print('SUNDAY')

#Zipped!

n,x=map(int,input().split())
Y=[]
for i in range(x):
    A=list(map(float,input().split()))
    Y.append(A)
YY=zip(*Y)
for i in YY:
    print(sum(i)/x)

#Nested Lists

if __name__ == '__main__':
    n=int(input())
    lista=[]
    voti=[]
    p=[]
    for i in range(n):
        nome=input()
        voto=float(input())
        voti=voti+[voto]
        lista+=[[nome,voto]]
    for i in range(n):
        if lista[i][1]==sorted(set(voti))[1]:
            p=p+[lista[i][0]]
print('\n'.join(sorted(p)))

#Finding the percentage

from decimal import Decimal
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    voti=student_marks[query_name]
    r=sum(voti)/len(voti)
print("%.2f" %r)

#Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        c=input().split()
        if c[0]=="print":
            print(l)
        elif c[0]=="insert":
            l.insert(int(c[1]),int(c[2]))
        elif c[0]=="remove":
            l.remove(int(c[1]))
        elif c[0]=="append":
            l.append(int(c[1]))
        elif c[0]=="pop":
            l.pop()
        elif c[0]=="sort":
            l.sort()
        elif c[0]=="reverse":
            l.reverse()
#Tuples 

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))

#Athlete Sort

N,M=map(int,input().split())
T=[]
for i in range(N):
    A=input()
    T.append(A)
k=int(input())
for r in sorted(T, key=lambda r: int(r.split()[k])):
    print(r)

#Arrays

def arrays(arr):
    arr.reverse()
    a=numpy.array(arr, float)
    return a

#Shape and Reshape

import numpy
a=input().split()
arr=numpy.array(a,int)
print(numpy.reshape(arr,(3,3)))

#Transpose and Flatten

import numpy
n,m=map(int,input().split())
arr=numpy.array([input().split() for i in range(n)],int)
print(numpy.transpose(arr))
print(arr.flatten())

#Concatenate

import numpy
n,m,p=map(int,input().split())
arrn=numpy.array([input().split() for i in range(n)],int)
arrm=numpy.array([input().split() for i in range(m)],int)
print(numpy.concatenate((arrn,arrm), axis= 0))

#Zeros and Ones

import numpy
dim=tuple(map(int,input().split()))
print(numpy.zeros((dim),int))
print(numpy.ones((dim),int))

#Eye and Identity

import numpy
numpy.set_printoptions(sign=' ') #found this line in the Discussion because there is a mismatch with the output due to the spacing in the solution
n,m=map(int,input().split())
print(numpy.eye(n,m))

#Array Mathematics

import numpy
n,m=map(int,input().split())
a= numpy.array([input().split() for _ in range(n)],int) # found this "for" solution
b= numpy.array([input().split() for _ in range(n)],int) # in the Discussion
print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(numpy.floor_divide(a,b))
print(numpy.mod(a,b))
print(numpy.power(a,b))

#Floor, Ceil and Rint

import numpy
numpy.set_printoptions(sign=' ') #same spacing problem as in the previous exercise 
a=numpy.array(input().split(),float)
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#Sum and Prod

import numpy
n,m=map(int,input().split())
a=numpy.array([input().split() for _ in range(n)],int)
print(numpy.prod(numpy.sum(a, axis=0), axis=0))

#Min and Max

import numpy
n,m= map(int, input().split())
a=numpy.array([input().split() for _ in range(n)],int)
print(numpy.max(numpy.min(a, axis=1), axis=0))

#Mean, Var, and Std

import numpy
n,m=map(int, input().split())
numpy.set_printoptions(legacy='1.13') #from the discussion
a=numpy.array([input().split() for _ in range(n)],int)
print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(numpy.std(a))

#Dot and Cross

import numpy
n=int(input())
a=numpy.array([input().split() for _ in range(n)],int)
b=numpy.array([input().split() for _ in range(n)],int)
print(numpy.dot(a,b))

#Inner and Outer

import numpy
a=numpy.array(input().split(),int)
b=numpy.array(input().split(),int)
print(numpy.inner(a,b))
print(numpy.outer(a,b))

#Polynomials

import numpy
a=numpy.array(input().split(),float)
x=float(input())
print(numpy.polyval(a, x))

#Linear Algebra

import numpy
n=int(input())
a=numpy.array([input().split() for _ in range(n)],float)
print(round(numpy.linalg.det(a),2))

#collections.Counter()

from collections import Counter
x=int(input())
s=list(map(int,input().split()))
n=int(input())
money=0
for _ in range(n):
    m,p=map(int,input().split())
    for size in set(s):
        if size==m and Counter(s)[size]!=0:
            money+=p
            s.remove(size)

print(money)

#DefaultDict Tutorial

from collections import defaultdict
d=defaultdict(list)
n,m=map(int,input().split())
for i in range(n):
    d[input()].append(str(i+1))    
for i in range(m):
    print (' '.join(d[input()]) or -1)

#Collections.namedtuple()

from collections import namedtuple
n=int(input())
vv=[]
vv.extend(input().split())
ss=namedtuple('ss', vv)
somma=0
for i in range(n):
    s=ss(*input().split())
    somma+=int(s.MARKS)
print(somma/n)

#Collections.OrderedDict()

from collections import OrderedDict
n=int(input())
d=OrderedDict()
for i in range(n):
    prod=list(input().split())
    price=prod[-1]
    prod.pop()
    if ' '.join(prod) not in d:
        d[' '.join(prod)]=int(price)
    else:
        d[' '.join(prod)]+=int(price)
for e in d:
    print(e, d[e])
#Word Order

from collections import OrderedDict
n=int(input())
w=OrderedDict()
for i in range(n):
    word=input()
    w.setdefault(word,0)
    w[word]+=1
print(len(w))
print(*w.values())

#Collections.deque()

from collections import deque
d=deque()
n=int(input())
for _ in range(n):
    c=list(input().split())
    if c[0]=="append":
        d.append(int(c[1]))
    elif c[0]=="pop":
        d.pop()
    elif c[0]=="appendleft":
        d.appendleft(int(c[1]))
    elif c[0]=="popleft":
        d.popleft()
print(*d) 

#Company Logo

#!/bin/python3
from collections import Counter
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    s = input()
    c=Counter(sorted(s)).most_common(3)
    for l in c:
        print(*l)
    
#Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    from collections import Counter
    return Counter(candles)[max(candles)]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Detect Floating Point Number

import re
n=int(input())
for _ in range(n):
    print (bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input()))) 
    #i've used an help from the discussion and the tutorial

#Re.split()

regex_pattern = r"[\.,\,]"	# Do not delete 'r'.

#Group(), Groups() & Groupdict()

import re
r=re.search(r'([a-zA-Z0-9])\1', input())
if r==None:
    print(-1)
else:
    print(r.group(1))    

#Re.findall() & Re.finditer()

import re
c='bcdfghjklmnpqrstvwxyz'
v='aeiou'
r='(?<=['+c+'])(['+v+']{2,})['+c+']'
contr=re.findall(r,input(),re.IGNORECASE)
if contr:
    print("\n".join(contr))
else:
    print("-1")

#Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    sh=5
    lk=2
    cu=2
    for i in range(n-1):
        sh=(sh//2)*3
        lk=sh//2
        cu+=lk
    return cu

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Re.start() & Re.end()

import re
s = input()
k = input()
if k in s:
    print(*[(i.start(), (i.start()+len(k)-1)) for i in re.finditer(r'(?={})'.format(k),s)], sep='\n')
else:
    print('(-1, -1)')

#Regex Substitution

import re
l=int(input())
for _ in range(l):
    print(re.sub('(?<=\s)\&\&\s','and ',re.sub('\s\|\|\s',' or ',input())))

#Validating Roman Numerals

regex_pattern = r"M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3}$)"	# Do not delete 'r'.

#Validating phone numbers

import re
n=int(input())
for _ in range(n):
    if re.match(r'[789]\d{9}$',input()):   
        print('YES')  
    else:  
        print('NO')

#Validating and Parsing Email Addresses

import re
n=int(input())
for _ in range(n):
    nome,email=input().split()
    if re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>',email):
        print(nome,email)

#Hex Color Code

import re
n=int(input())
for _ in range(n):
    l=re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if l:
        print(*l, sep='\n')

#Validating UID 

import re
n=int(input())
for _ in range(n):
    if re.match(r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[A-Za-z0-9]{10}$',input()):
        print('Valid') 
    else:
        print('Invalid')

#Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    a=k*sum(map(int,n))
    if a%9==0:
        return 9
    else:
        return a%9

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
    
#Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    u=arr[-1]
    while (u<arr[n-2]) and (n-2>=0):
        arr[n-1]=arr[n-2]
        print(' '.join(map(str,arr)))
        n-=1
    arr[n-1]=u
    print(' '.join(map(str,arr)))

if __name__ == '__main__':
    n = int(input())
    

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(1,n):
        p=arr[i]
        pp=i-1
        while pp>=0 and p<arr[pp]:
            arr[pp+1]=arr[pp]
            pp-=1
        arr[pp+1]=p
        print(' '.join(map(str,arr)))
    


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

#Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if (v2-v1)!=0 and (x1-x2)/(v2-v1)>0 and (x1-x2)%(v2-v1)==0:
        return 'YES'
    else:
        return 'NO'
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Capitalize!

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(s):

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

