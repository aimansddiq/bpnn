import math
X = int(input("Number of X: "))
Z = int(input("Number of Z: "))
Y = int(input("Number of Y: "))
lr = float(input("Learning Rate: "))
x_values = []
vij = []
wjk = []
t = []

b_z = []
b_y = []
z_in = []
z_value = []
y_in = []
y_value = []
delta_k = []
wc = []
vc = []
wb = []
vb = []
delta_z_in =[]
delta_z_value = []
v_new = []
w_new = []
wb_new = []
vb_new = []

for i in range(X):
    w = float(input("X{} values: ".format(i+1)))
    x_values.append(w)
for i in range(X*Z):
    w = float(input("Weight of vij{} : ".format(i+1)))
    vij.append(w)
for i in range(Z):
    w = float(input("Bias Z {} : ".format(i+1)))
    b_z.append(w)
for i in range(Z*Y):
    w = float(input("Weight of wjk{} : ".format(i+1)))
    wjk.append(w)
for i in range(Y):
    w = float(input("Bias Y {} : ".format(i+1)))
    b_y.append(w)
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
    if(b_z[i] > 0):
        calculation += "({} * {}) +".format(1,b_z[i])
        z_in[i] += 1 * b_z[i]
    print(calculation[:-1])
    print("Z{}-in = {:.4f}".format(i+1,z_in[i]))
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
    if(b_y[i] > 0):
        calculation += "({} * {}) +".format(1,b_y[i])
        y_in[i] += 1 * b_y[i]
    print(calculation[:-1])
    print("Y{}-in = {:.4f}".format(i+1,y_in[i]))
    y_value[i] = 1/(1+math.exp(-y_in[i]))
    print("Y{} = 1/(1+e^{:.4f})".format(i+1,-y_in[i]))
    print("Y{} = {:.4f}\n".format(i+1,y_value[i]))

print("2. BACKPROPAGATE THE ERROR PHASE\nStep 6:")
print("delta-kj = (t-y)(y)(1-y)")
for i in range(Y):
    delta_k.append((t[i] - y_value[i])*(y_value[i])*(1-y_value[i]))
    print("delta-k{0} = ({1} - {2:.4f})*({2:.4f})*(1-{2:.4f})".format(i+1,t[i],y_value[i]))
    print("delta-k{} = {:.4f}".format(i+1,delta_k[i]))

print("\nWEIGHT CHANGES (∆w)")
print("(delta-k)(lr)(Zj)\n")
for i in range(Z):
    for j in range(Y):        
        wc.append(delta_k[j]*lr*z_value[i])
        print("∆w{}{} = ({:.4f})*({})*({:.4f}) = {:.4f}".format(i+1,j+1,delta_k[j],lr,z_value[i],wc[-1]))
for j in range(len(b_y)):
    if(b_y[j] > 0):        
        wb.append(delta_k[j]*lr*1)
        print("∆wb{}{} = ({:.4f})*({})*({:.1f}) = {:.4f}".format(0,j+1,delta_k[j],lr,1,wb[-1]))
    else:
        wb.append(0)

print("\nStep 7:")
print("delta Zj-in = sum(delta-k * wjk)")
for i in range(Z):
    delta_z_in.append(0)
    calculation = "delta Z{}-in = ".format(i+1)
    for j in range(Y):
        delta_z_in[i] += (delta_k[j] * wjk[(Y*i)+j])
        calculation += "({:.4f} * {:.2f}) + ".format(delta_k[j],wjk[(Y*i)+j])
    print(calculation[:-2])
    print("delta Z{}-in = {:.4f}".format(i+1,delta_z_in[i]))
    delta_z_value.append(delta_z_in[i] * z_value[i] * (1-z_value[i]))
    print("delta Z{0} = ({1:.4f})*({2:.4f})*(1-{2:.4f})".format(i+1,delta_z_in[i],z_value[i]))
    print("delta Z{} = {:.4f}\n".format(i+1,delta_z_value[i]))

print("\nWEIGHT CHANGES (∆v)")
print("(delta-j)(lr)(Xj)\n")
for i in range(X):
    for j in range(Z):
        vc.append(delta_z_value[j]*lr*x_values[i])
        print("∆v{}{} = ({:.4f})*({})*({:.2f}) = {:.4f}".format(i+1,j+1,delta_z_value[j],lr,x_values[i],vc[-1]+0))
for j in range(Z):
    if(b_z[j] > 0):        
        vb.append(delta_z_value[j]*lr*1)
        print("∆vb{}{} = ({:.4f})*({})*({:.1f}) = {:.4f}".format(0,j+1,delta_z_value[j],lr,1,vb[-1]+0))
    else:
        vb.append(0)

print("\n3. WEIGHT UPDATES\n\nStep 8:")
for i in range(len(vij)):
    v_new.append(vij[i] + vc[i])
    print("v(new) = {} + {:.4f} = {:.4f}".format(vij[i],vc[i],v_new[-1]))
for i in range(len(b_z)):
    if(b_z[i] > 0):
        vb_new.append(b_z[i] + vb[i])
        print("vb(new) = {} + {:.4f} = {:.4f}".format(b_z[i],vb[i],vb_new[-1]))
for i in range(len(wjk)):
    w_new.append(wjk[i] + wc[i])
    print("w(new) = {} + {:.4f} = {:.4f}".format(wjk[i],wc[i], w_new[-1]))
for i in range(len(b_y)):
    if(b_y[i] > 0):
        wb_new.append(b_y[i] + wb[i])
        print("wb(new) = {} + {:.4f} = {:.4f}".format(b_y[i],wb[i], wb_new[-1]))
