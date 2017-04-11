'''
trust and distrust propagation
see Guha et al. WWW 2004 for more information
'''

from sklearn.linear_model import LogisticRegression
import time

src_lst = []
dest_lst = []
sign_lst = []

def load_data (filename):
    with open (filename, 'r') as f:
        skip (f)
        for line in f:
            src,dst,sign = map (int, line.split()[0:3])
            src_lst.append(src)
            dest_lst.append(dst)
            sign_lst.append(lst)

def predict_guha (src_id, dst_id):
    assert (len(src_lst) == len(dest_lst) == len(sign_lst))
    N = len (src_lst)
    '''
        predict sign of the link from src_id --> dst_id
    '''
    start_time = time.time ()
    # direct propagation: friend of friend is friend
    for i in range (N):
        if src_lst[i] == src_id and dest_lst[i] != dst_id and sign_lst[i] == 1:
            for j in range (N):
                if src_lst[j] == dest_lst[i] and src_lst[j] == 1:
                    return 1
    # co-citation:
    for i in range (N):
        if src_lst[i] != src_id and dest_lst[i] == dst_id:
            indices1 = [j for j, x in enumerate(src_lst) if x == src_lst[i]]
            indices2 = [j for j, x in enumerate(src_lst) if x == src_id]
            target1 = dest_lst[indices1]
            target2 = dest_lst[indices2]
            target1.remove (dst_id)
            target2.remove (dst_id)
            if set(target1).isdisjoint(target2):
                return 1

    # transpose trust
    for i in range (N):
        if src_lst[i] != src_id and dest_lst[i] == dst_id:
            for j in range (N):
                if src_lst[j] == src_id and dest_lst[j] == src_lst[i]:
                    return 1

    # trust coupling
    for i in range (N):
        if src_lst[i] == src_id and dest_lst[i] != dst_id:
            for j in range (N):
                if src_lst[j] == dest_lst[i] and src_lst[j] != sign_lst[i]:
                    return -1

    end_time = time.time ()
    with open ("time_guha.txt", "a") as f:
        f.write (str (start_time - end_time) + '\n')
    
    return (-1)

def get_features_leskovec (u,v):
    assert (len(src_lst) == len(dest_lst) == len(sign_lst))
    feature_list = []
    d_u_in_plus = 0
    d_v_in_minus = 0
    d_u_out_plus = 0
    d_v_out_minus = 0

    C_u_v = 0
    d_out_u = 0
    d_out_v = 0
    N = len (src_lst)

    for i in range (N):
        if dest_lst[i] == u and sign_lst[i] == 1:
            d_u_in_plus += 1
        if dest_lst[i] == v and sign_lst[i] == -1:
            d_v_in_minus += 1

        if src_lst[i] == u and sign_lst[i] == 1:
            d_u_out_plus += 1
        if src_lst[i] == v and sign_lst[i] == -1:
            d_v_out_minus += 1

        if (src_lst[i] == u and dest_lst[i] != v) or (src_lst[i] != u and dest_lst[i] == v) \
            or (src_lst[i] == v and dest_lst[i] != u) or (src_lst[i] != v and dest_lst[i] == u):
            C_u_v += 1

        if src_lst[i] == u and dest_lst[i] != v:
            d_out_u += 1

        if src_lst[i] == v and dest_lst[i] != u:
            d_out_v += 1

    MAX_ID = max (max(src_lst), max (dest_lst))

    triangle_list = [0] * 16
    for i in range (N):
        w = -1
        # dire = 0 --> w to u,v
        dire = 0
        # target = 0: found w to u
        targe = 0 
        if src_lst[i] == u and dest_lst[i] != v:
            w = dest_lst[i]
            dire = 1

        if src_lst[i] == v and dest_lst[i] != u:
            w = dest_lst[i]
            targe = 1
            dire = 1

        if src_lst[i] != u and dest_lst[i] == v:
            w = src_lst[i]
            targe = 1

        if src_lst[i] != v and dest_lst[i] == u:
            w = src_lst[i]

        for j in range(N):
            if src_lst[j] == w:
                if targe == 0 and dest_lst[j] == v:
                    triangle_list[dire*8 + sign_lst[i]*4 + 0*2 + sign_lst[j]] += 1
                    continue

            if src_lst[j] == w:
                if targe == 1 and dest_lst[j] == u:
                    triangle_list[dire*2 + sign_lst[i] + 0*8 + sign_lst[j]*4] += 1

            if dest_lst[j] == w:
                if targe == 0 and src_lst[j] == v:
                    triangle_list[dire*8 + sign_lst[i]*4 + 1*2 + sign_lst[j]] += 1
                    continue

            if dest_lst[j] == w:
                if targe == 1 and src_lst[j] == u:
                    triangle_list[dire*2 + sign_lst[i] + 1*8 + sign_lst[j]*4] += 1
                    continue

    feature_list.append (d_u_in_plus)
    feature_list.append (d_v_in_minus)
    feature_list.append (d_u_out_plus)
    feature_list.append (d_v_out_minus)
    feature_list.append (C_u_v)
    feature_list.append (d_out_u)
    feature_list.append (d_out_v)

    feature_list = feature_list + triangle_list

    return feature_list

def predict_leskovec (src_id, dst_id):
    N = len (src_lst)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    times = []

    test_id = 0
    start_time = time.time ()
    for i in range (N):
        if src_lst[i] != src_id and dest_lst[i] != dst_id:
            X_train.append (get_features_leskovec(src_lst[i],dest_lst[i]))
            Y_train.append (sign_lst[i])
        else:
            test_id = i
    X_test.append (get_features_leskovec(src_id, dst_id))
    Y_test.append (sign_lst[test_id])
    end_time = time.time ()
    times.append (end_time - start_time)

    start_time = time.time ()

    try:
        model = LogisticRegression()
        model.fit(X_train, Y_train)

        pred = model.predict (X_test)
    except Exception, e:
        pred = 1
    end_time = time.time ()
    times.append (end_time - start_time)

    with open ("time_leskovec.txt", "a") as f:
        f.write (str(test_id) + '\t' + str(times[0]) + '\t' + str(times[1]) + '\n')
    return (pred, sign_lst[test_id])

def main ():
    with open ("time_guha.txt", 'w') as f:
        f.write("Time\n")

    with open ("time_leskovec.txt", 'w') as f:
        f.write("ID\tfeature\tlearn\n")

    with open ("acc_guha.txt", 'w') as f:
        f.write("accuracy\n")

    with open ("acc_lesk.txt", 'w') as f:
        f.write("accuracy\n")

    data_filename = "../data/wikipedia.csv"

    data_cnt = 0
    pred_cnt = 0

    acc_guha = 0.0
    acc_lesk = 0.0

    with open (data_filename, 'r') as data_file:
        next (data_file)
        for line in data_file:
            content = line.split (',')
            trustor = content [1].replace ('\"','')
            trustee = content [2].replace ('\"','')
            sign = content [3].replace ('\"','')

            src_lst.append(int(trustor))
            dest_lst.append (int(trustee))
            sign_lst.append(int (sign))

            data_cnt += 1
            if data_cnt > 100:
                print ('Predict line ' + str(data_cnt))
                pred_guha = predict_guha (trustor, trustee)
                pred_lesk = predict_leskovec (trustor, trustee)
                if pred_guha == int (sign):
                    acc_guha += 1
                if int (pred_lesk[0]) == int (sign):
                    print ('Lesk true')
                    acc_lesk += 1
                pred_cnt += 1
                print (str(pred_guha) + ',' + str (pred_lesk[0]) + ',' + str(sign) + ',' + str(pred_cnt))
                with open ("acc_guha.txt", 'a') as f:
                    f.write (str(float(acc_guha) / pred_cnt) + '\n')
                with open ("acc_lesk.txt", 'a') as f:
                    f.write (str(float(acc_lesk) / pred_cnt) + '\n')

if __name__ == '__main__':
    main()