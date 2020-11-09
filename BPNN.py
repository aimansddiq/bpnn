import math
X = int(input("Number of X: "))
Z = int(input("Number of Z: "))
Y = int(input("Number of Y: "))
lr = float(input("Learning Rate: "))
x_values = []
vij = []
wjk = []
t = []

z_in = []
z_value = []
y_in = []
y_value = []
delta_k = []
wc = []
vc = []
delta_z_in =[]
delta_z_value = []
v_new = []
w_new = []

for i in range(X):
    w = float(input("X{} values: ".format(i+1)))
    x_values.append(w)
for i in range(X*Z):
    w = float(input("Weight of vij{} : ".format(i+1)))
    vij.append(w)
for i in range(Z*Y):
    w = float(input("Weight of wjk{} : ".format(i+1)))
    wjk.append(w)
for i in range(Y):
    w = float(input("T values{} : ".format(i+1)))
    t.append(w)

print('/'*100)
print("\n1. FEEDFORWARD PHASE\n\nStep 4:")
for i in range(Z):
    z_in.append(0)
    z_value.append(0)
    calculation = "Z{}-in = ".format(i+1)
    for j in range(X):
        calculation += "({} * {}) +".format(x_values[j],vij[((j*Z)+i)])
        z_in[i] += x_values[j] * vij[((j*Z)+i)]
    print(calculation[:-1])
    print("Z{}-in = {}".format(i+1,z_in[i]))
    z_value[i] = 1/(1+math.exp(-z_in[i]))
    print("Z{} = 1/(1+e^{:.4f})".format(i+1,-z_in[i]))
    print("Z{} = {:.4f}\n".format(i+1,z_value[i]))

print("Step 5:")
for i in range(Y):
    y_in.append(0)
    y_value.append(0)
    calculation = "Y{}-in = ".format(i+1)
    for j in range(Z):
        calculation += "({:.4f} * {}) +".format(z_value[j],wjk[((j*Y)+i)])
        y_in[i] += z_value[j] * wjk[((j*Y)+i)]
    print(calculation[:-1])
    print("Y{}-in = {:.4f}".format(i+1,y_in[i]))
    y_value[i] = 1/(1+math.exp(-y_in[i]))
    print("Y{} = 1/(1+e^{:.4f})".format(i+1,-y_in[i]))
    print("Y{} = {:.4f}\n".format(i+1,y_value[i]))

print("2. BACKPROPAGATE THE ERROR PHASE\nStep 6:")
for i in range(Y):
    delta_k.append((t[i] - y_value[i])*(y_value[i])*(1-y_value[i]))
    print("delta-k{} = (t-y)(y)(1-y)".format(i+1))
    print("delta-k{0} = ({1} - {2:.4f})*({2:.4f})*(1-{2:.4f})".format(i+1,t[i],y_value[i]))
    print("delta-k{} = {:.4f}".format(i+1,delta_k[i]))

print("\nWEIGHT CHANGES (wc)")
print("(delta-k)(lr)(Zj)\n")
for i in range(Z):
    for j in range(Y):
        print("wc{}{} = ({:.4f})*({})*({:.4f})".format(i+1,j+1,delta_k[j],lr,z_value[i]))
        wc.append(delta_k[j]*lr*z_value[i])
        print("wc{}{} = {:.4f}".format(i+1,j+1,wc[-1]))

print("\nStep 7:")
print("delta Zj-in = sum(delta-k * wjk)")
for i in range(Z):
    delta_z_in.append(0)
    calculation = "delta Z{}-in = ".format(i+1)
    for j in range(Y):
        delta_z_in[i] += (delta_k[j] * wjk[(2*i)+j])
        calculation += "({:.4f} * {:.2f}) + ".format(delta_k[j],wjk[(2*i)+j])
    print(calculation[:-1])
    print("delta Z{}-in = {:.4f}".format(i+1,delta_z_in[i]))
    delta_z_value.append(delta_z_in[i] * z_value[i] * (1-z_value[i]))
    print("delta Z{0} = ({1:.4f})*({1:.4f})*(1-{1:.4f})".format(i+1,-delta_z_in[i]))
    print("delta Z{} = {:.4f}\n".format(i+1,delta_z_value[i]))

print("\nWEIGHT CHANGES (vc)")
print("(delta-j)(lr)(Xj)\n")
for i in range(X):
    for j in range(Z):
        print("vc{}{} = ({:.4f})*({})*({:.4f})".format(i+1,j+1,delta_z_value[j],lr,x_values[i]))
        vc.append(delta_z_value[j]*lr*x_values[i])
        print("vc{}{} = {}".format(i+1,j+1,round(vc[-1]+0,4)))

print("\n3. WEIGHT UPDATES\n\nStep 8:")
for i in range(len(vij)):
    v_new.append(vij[i] + vc[i])
    print("v(new) = {} + {:.4f}".format(vij[i],vc[i]))
    print("v(new) = {}".format(round(v_new[-1],4)))

print("\n")
for i in range(len(wjk)):
    w_new.append(wjk[i] + wc[i])
    print("w(new) = {} + {:.4f}".format(wjk[i],wc[i]))
    print("w(new) = {}".format(round(w_new[-1],4)))