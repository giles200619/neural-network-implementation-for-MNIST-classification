
import numpy as np
import math
import random
import main_functions as main



def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    shuffule_img = np.zeros(im_train.shape)
    shuffle_label = np.zeros(label_train.shape)
    order = random.sample(range(0, im_train.shape[1]),im_train.shape[1])
    for i in range(0, len(order)):
        shuffule_img[:,i]=im_train[:,order[i]]
        shuffle_label[:,i]=label_train[:,order[i]]
#    idx = np.random.rand(*im_train.shape).argsort(1)
#    shuffule_img = np.take_along_axis(im_train,idx,axis=1)
#    shuffle_label = np.take_along_axis(label_train,idx,axis=1)
#    mini_batch_x = []
#    mini_batch_y = []
#    for i in range(0,math.ceil(np.asarray(shuffule_img).shape[1]/batch_size)):
#       mini_batch_x.append(shuffule_img[:,32*i:32*i+32])
#       #shuffle_label[:,32*i:32*i+32]
#       batch_label = np.zeros((10,batch_size))
#       for j in range(0,batch_size):
#           batch_label[shuffle_label[0][32*i+j]][j]=1
#       mini_batch_y.append(batch_label)
    mini_batch_x = []
    mini_batch_y = []
    for i in range(0,math.ceil(shuffule_img.shape[1]/batch_size)):
        batchx = shuffule_img[:,i*32:i*32+32]
        mini_batch_x.append(batchx)
        batch_label = np.zeros((10,batch_size))
        for j in range(0,batch_size):
            batch_label[int(shuffle_label[0][32*i+j])][j]=1
        mini_batch_y.append(batch_label)
    return mini_batch_x, mini_batch_y

# =============================================================================
# for i in range(0,12000):
#     if (mini_batch_x[11][:,3] == im_train[:,i]).all():
#         for j in range(0,10):
#             if mini_batch_y[11][j][3]==1:
#                 if j == label_train[:,i]:
#                     print("get")
# =============================================================================

def fc(x, w, b):
    # TO DO
    y = np.add(np.matmul(w,x),b)
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = np.matmul(np.transpose(w),dl_dy)
    dl_dw = np.matmul(dl_dy,np.transpose(x))
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.square(np.subtract(y_tilde,y))
    dl_dy = (2)*np.subtract(y_tilde,y)
    #dl_dy = np.transpose(dl_dy)
    return l, dl_dy

def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate=0.001
    decay_rate=0.5
    w = np.random.normal(0,2,1960)
    dif = max(w)-min(w)
    min_w = min(w)
    for i in range(0,len(w)):
        w[i] = (w[i]-min_w)/dif
    w = np.reshape(w,(10,196))
    b = np.random.randn(10,1)
    k = 0
    for iIter in range(0,100000):
        if iIter%2000 == 0:
            learning_rate = learning_rate*decay_rate
        dL_dw = 0
        dL_db = 0
        Loss = 0
        for x in range(0,32):
            y_tilde = fc(np.reshape(mini_batch_x[k][:,x],(196,1)),w,b)
            l, dl_dy = loss_euclidean(y_tilde, np.reshape(mini_batch_y[k][:,x],(10,1)))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, np.reshape(mini_batch_x[k][:,x],(196,1)), w, b, np.reshape(mini_batch_y[k][:,x],(10,1)))
            dL_dw = np.add(dL_dw, dl_dw)
            dL_db = np.add(dL_db, dl_db)
            Loss = Loss + l
        print(iIter, np.linalg.norm(Loss)/32)
        k+=1
        if k==len(mini_batch_x):
            k = 0
        w = np.subtract(w, (learning_rate/32)*dL_dw)
        b = np.subtract(b, (learning_rate/32)*dL_db)
    
    return w, b


def loss_cross_entropy_softmax(x, y):
    # TO DO
    yi = np.zeros((x.shape))
    exps = np.exp(x)
    yi = exps / np.sum(exps)
# =============================================================================
#    for i in range(0,yi.shape[0]):
#        esum+=math.exp(x[i][0])
#    for i in range(0,yi.shape[0]):
#        yi[i][0] = math.exp(x[i][0])/esum
# =============================================================================
    l = 0
    for i in range(0,yi.shape[0]):
        l+=-y[i][0]*np.log(yi[i][0])
    #dl_dyi = np.divide(y,yi)/math.log(10)
    #dl_dy = np.multiply(dl_dyi,np.multiply(yi,1-yi))
    #dl_dy = np.add(np.subtract(yi,y),np.multiply(yi,1-yi))
    #dl_dy = dl_dx = dl_dyi * dyi_dx
    dl_dy = np.subtract(yi,y)
    
    return l, dl_dy

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate=1
    decay_rate=0.5
    w = np.random.normal(0,2,1960)
    dif = max(w)-min(w)
    min_w = min(w)
    for i in range(0,len(w)):
        w[i] = (w[i]-min_w)/dif
    w = np.reshape(w,(10,196))
    b = np.random.randn(10,1)
    k = 0
    for iIter in range(0,10000):
        if iIter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dL_dw = 0
        dL_db = 0
        Loss = 0
        for x in range(0,32):
            a1 = fc(np.reshape(mini_batch_x[k][:,x],(196,1)),w,b)
            l, dl_da1 = loss_cross_entropy_softmax(a1, np.reshape(mini_batch_y[k][:,x],(10,1)))
            dl_dx, dl_dw, dl_db = fc_backward(dl_da1, np.reshape(mini_batch_x[k][:,x],(196,1)), w, b, np.reshape(mini_batch_y[k][:,x],(10,1)))
            dL_dw = np.add(dL_dw, dl_dw)
            dL_db = np.add(dL_db, dl_db)
            Loss = Loss + l
        print(iIter, (Loss)/32)
        k+=1
        if k==len(mini_batch_x):
            k = 0
        w = np.subtract(w, (learning_rate/32)*dL_dw)
        b = np.subtract(b, (learning_rate/32)*dL_db)
    
    return w, b


def relu(x):
    # TO DO
    y = x
    if len(x.shape) == 2:
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                if x[i][j]<0:
                    y[i][j]=0
    if len(x.shape) == 3:
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                for k in range(0,x.shape[2]):
                    if x[i][j][k]<0:
                        y[i][j][k]=0
    
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy
    if len(x.shape) == 2:
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                if x[i][j]<0:
                    dl_dx[i][j]=0
    if len(x.shape) == 3:
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                for k in range(0,x.shape[2]):
                    if x[i][j][k]<0:
                        dl_dx[i][j][k]=0
    return dl_dx

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate=0.05
    decay_rate=0.9
    w1 = np.random.normal(0,1,196*30)
    dif = max(w1)-min(w1)
    min_w1 = min(w1)
    for i in range(0,len(w1)):
        w1[i] = (w1[i]-min_w1)/dif
    w1 = np.reshape(w1,(30,196))
    w2 = np.random.normal(0,1,300)
    dif = max(w2)-min(w2)
    min_w2 = min(w2)
    for i in range(0,len(w2)):
        w2[i] = (w2[i]-min_w2)/dif
    w2 = np.reshape(w2,(10,30))    
    b1 = np.random.randn(30,1)
    b2 = np.random.randn(10,1)
    k = 0
    for iIter in range(0,100000):
        if iIter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dL_dw1 = 0
        dL_db1 = 0
        dL_dw2 = 0
        dL_db2 = 0
        Loss = 0
        for x in range(0,32):
            
            a1 = fc(np.reshape(mini_batch_x[k][:,x],(196,1)),w1,b1)
            rel1 = relu(a1)
            #rel1 = a1#
            a2 = fc(rel1,w2,b2)
            rel2 = relu(a2)
            #rel2 = a2#            
            l, dl_dy = loss_cross_entropy_softmax(rel2,np.reshape(mini_batch_y[k][:,x],(10,1)))
            dl_dx = relu_backward(dl_dy,a2,rel2)
            #dl_dx = dl_dy#
            dl_da1, dl_dw2, dl_db2 = fc_backward(dl_dx, a1, w2, b2, a2)
            dl_dx = relu_backward(dl_da1,a1,rel1)
            #l_dx = dl_da1#
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dx, np.reshape(mini_batch_x[k][:,x],(196,1)), w1, b1, a1)
            
            dL_dw1 = np.add(dL_dw1, dl_dw1)
            dL_db1 = np.add(dL_db1, dl_db1)
            dL_dw2 = np.add(dL_dw2, dl_dw2)
            dL_db2 = np.add(dL_db2, dl_db2)
            Loss = Loss + l
        print(iIter, (Loss)/32)
        k+=1
        if k==len(mini_batch_x):
            k = 0
        w1 = np.subtract(w1, (learning_rate/32)*dL_dw1)
        b1 = np.subtract(b1, (learning_rate/32)*dL_db1)
        w2 = np.subtract(w2, (learning_rate/32)*dL_dw2)
        b2 = np.subtract(b2, (learning_rate/32)*dL_db2)
    
    
    
    return w1, b1, w2, b2




def conv(x, w_conv, b_conv):
    # TO DO
    y = np.zeros((x.shape[0],x.shape[1],3))
    x0 = np.pad(x, ((1,1),(1,1)), 'constant')
    for i in range(0,y.shape[0]):
        for j in range(0,y.shape[1]):
            for m in range(0,3):
                for n in range(0,3):
                    y[i][j][0]+=w_conv[m][n][0][0]*x0[m+i][n+j]
                    y[i][j][1]+=w_conv[m][n][0][1]*x0[m+i][n+j]
                    y[i][j][2]+=w_conv[m][n][0][2]*x0[m+i][n+j]
            y[i][j][0]+=b_conv[0]
            y[i][j][1]+=b_conv[1]
            y[i][j][2]+=b_conv[2]
            
    return y

def conv2(x, w_conv, b_conv):
    # TO DO
    y = np.zeros((x.shape[0],x.shape[1],3))
    x0 = np.pad(x, ((1,1),(1,1)), 'constant')
    for i in range(0,y.shape[0]):
        for j in range(0,y.shape[1]):
            for m in range(0,3):
                for n in range(0,3):
                    y[i][j][0]+=w_conv[m][n][0][0]*x0[m+i][n+j]
                    y[i][j][1]+=w_conv[m][n][0][1]*x0[m+i][n+j]
                    y[i][j][2]+=w_conv[m][n][0][2]*x0[m+i][n+j]
            y[i][j][0]+=b_conv[0]
            y[i][j][1]+=b_conv[1]
            y[i][j][2]+=b_conv[2]
            
    return y

#test = np.reshape(test,(14,14),order='F')
#test = np.random.randn(14,14,3)
#test2 = test[:,:,0]

def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)
    for k in range(0,3):
        dl_dw[0][0][0][k] = np.sum(np.multiply(x[0:13,0:13],dl_dy[1:14,1:14,k]))
        dl_dw[1][0][0][k] = np.sum(np.multiply(x[0:14,0:13],dl_dy[0:14,1:14,k]))
        dl_dw[2][0][0][k] = np.sum(np.multiply(x[1:14,0:13],dl_dy[0:13,1:14,k]))
        dl_dw[0][1][0][k] = np.sum(np.multiply(x[0:13,0:14],dl_dy[1:14,0:14,k]))
        dl_dw[1][1][0][k] = np.sum(np.multiply(x[:,:],dl_dy[:,:,k]))
        dl_dw[2][1][0][k] = np.sum(np.multiply(x[1:14,0:14],dl_dy[0:13,0:14,k]))
        dl_dw[0][2][0][k] = np.sum(np.multiply(x[0:13,1:14],dl_dy[1:14,0:13,k]))
        dl_dw[1][2][0][k] = np.sum(np.multiply(x[0:14,1:14],dl_dy[0:14,0:13,k]))
        dl_dw[2][2][0][k] = np.sum(np.multiply(x[1:14,1:14],dl_dy[0:13,0:13,k]))
    for n in range(dl_dy.shape[2]):    
        for i in range(dl_dy.shape[0]):
            for j in range(dl_dy.shape[1]):
                dl_db[n]+=dl_dy[i][j][n]
            
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    y = np.zeros((int(x.shape[0]/2),int(x.shape[1]/2),x.shape[2]))
    x_record = np.zeros(x.shape)
    for i in range(0,y.shape[0]):
        for j in range(0,y.shape[1]):
            for k in range(0,y.shape[2]):
                y[i][j][k]=max(x[2*i][2*j][k],x[2*i][2*j+1][k],x[2*i+1][2*j][k],x[2*i+1][2*j+1][k])
                for m in range(0,2):
                    for n in range(0,2):
                        if x[2*i+m][2*i+n][k] == y[i][j][k]:
                            x_record[2*i+m][2*i+n][k] = 1
    
    return y, x_record

def pool2x2_backward(dl_dy, x, y, x_record):
    # TO DO
    dl_dx = x_record
    for i in range(0,dl_dy.shape[0]):
        for j in range(0,dl_dy.shape[1]):
            for k in range(0,dl_dy.shape[2]):
                for m in range(0,2):
                    for n in range(0,2):
                        if x_record[2*i+m][2*i+n][k] == 1:
                            dl_dx[2*i+m][2*i+n][k] = dl_dy[i][j][k]
    
    return dl_dx


def flattening(x):
    # TO DO
    
    #y = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
    x = np.asarray(x)
    y = x.reshape(147,1)
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    #dl_dx = dl_dy.reshape(x.shape)
    
    dl_dx = np.reshape(dl_dy,(x.shape))
    return dl_dx


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate=0.1
    decay_rate=0.9
    w_conv = np.random.normal(0,1,27)
    dif = max(w_conv)-min(w_conv)
    min_w_conv = min(w_conv)
    for i in range(0,len(w_conv)):
        w_conv[i] = (w_conv[i]-min_w_conv)/dif
    w_conv = np.reshape(w_conv,(3,3,1,3))
    w_fc = np.random.normal(0,1,1470)
    dif = max(w_fc)-min(w_fc)
    min_w_fc = min(w_fc)
    for i in range(0,len(w_fc)):
        w_fc[i] = (w_fc[i]-min_w_fc)/dif
    w_fc = np.reshape(w_fc,(10,147))
    b_conv = np.random.randn(3)
    b_fc = np.random.randn(10,1)
    k = 0
    for iIter in range(0,10000):
        if iIter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dL_dw_conv = 0
        dL_db_conv = 0
        dL_dw_fc = 0
        dL_db_fc = 0
        Loss = 0
        for x in range(0,32):
            img = np.reshape(mini_batch_x[k][:,x],(196,1))
            img = np.reshape(img,(14,14),order='F')
            out_conv = conv(img, w_conv, b_conv)
            out_rel = relu(out_conv)
            out_pool,pool_record = pool2x2(out_rel)
            out_flat = flattening(out_pool)
            out_fc = fc(out_flat, w_fc, b_fc)
            relu2 = relu(out_fc)
            l, dl_dy = loss_cross_entropy_softmax(out_fc,np.reshape(mini_batch_y[k][:,x],(10,1)))
            drelu2 = relu_backward(dl_dy,out_fc,relu2)
            dfc, dl_dw_fc, dl_db_fc = fc_backward(drelu2,out_flat,w_fc, b_fc,out_fc)
            dflat = flattening_backward(dfc,out_pool,out_flat)
            dpool = pool2x2_backward(dflat,out_rel,out_pool,pool_record)
            drelu = relu_backward(dpool,out_conv,out_rel)
            dl_dw_conv,dl_db_conv = conv_backward(drelu,img,w_conv, b_conv,out_conv)
            
            dL_dw_conv = np.add(dL_dw_conv, dl_dw_conv)
            dL_db_conv = np.add(dL_db_conv, dl_db_conv)
            dL_dw_fc = np.add(dL_dw_fc, dl_dw_fc)
            dL_db_fc = np.add(dL_db_fc, dl_db_fc)
            Loss = Loss + l
        print(iIter, (Loss)/32)
        if Loss/32 < 0.01:
            break
        k+=1
        if k==len(mini_batch_x):
            k = 0
        w_conv = np.subtract(w_conv, (learning_rate/32)*dL_dw_conv)
        b_conv = np.subtract(b_conv, (learning_rate/32)*dL_db_conv)
        w_fc = np.subtract(w_fc, (learning_rate/32)*dL_dw_fc)
        b_fc = np.subtract(b_fc, (learning_rate/32)*dL_db_fc)
        
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    #sys.path.append('D:/document/work_UMN/2019Fall/Computer Vision/HW/HW4')
    #os.chdir('D:/document/work_UMN/2019Fall/Computer Vision/HW/HW4')
    #plt.imshow(mnist_train[’im_train’][:, 0].reshape((14, 14), order=’F’), cmap=’gray’)
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()


