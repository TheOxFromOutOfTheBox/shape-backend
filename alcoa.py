import math

def factorial1(number):
    result = 1
    for factor in range(2, number+1):
        result *= factor
    return 1.0/result

def RatioOfNV(column):
    ratio = [0]*len(column)
    quadraticSum = 0.0
    for i in range(len(column)):
        quadraticSum += column[i]*column[i]
        
    for j in range(len(column)):
        ratio[j] = column[j]/(math.pow(quadraticSum, 0.5))
        
    return ratio

def distanceValue(mu1, mu2):
    distanceValue = 0.0
    distanceValue = math.pow(math.pow(mu1-mu2, 2.0), 1.0/2.0)
    #distanceValue = math.pow(math.pow(abs(mu1-mu2), 2.0), 1.0/2.0)
    return distanceValue

def SelectionSort(dom):
    tempArray = [0]*len(dom)
    for k in range(len(dom)):
        tempArray[k] = dom[k]
        
    for i in range(len(tempArray)-1):
        indexOfMax = i
        for j in range(i+1, len(tempArray)):
            if tempArray[indexOfMax] < tempArray[j]:
                indexOfMax = j
                
        if indexOfMax != i:
            temp = tempArray[i]
            tempArray[i] = tempArray[indexOfMax]
            tempArray[indexOfMax] = temp
            
    return tempArray

def permutation(s, rs, res):
    if len(s) == 1:
        rs.append(s[0])
        tmp = [a for a in rs]
        res.append(tmp)
        rs.pop()
    else:
        for i in range(len(s)):
            rs.append(s[i])
            tmp = [a for a in s]
            tmp.pop(i)
            permutation(tmp, rs, res)
            rs.pop()


def FPWMSMHTT(delta,m,n,beta,ww,kk):
    q=1
    s2=[]
    rs=[]
    qp=[]
    for i in range(n):
        s2.append(i)
    permutation(s2,rs,qp)
    pp=[]
    for j in range(0,kk):
        pp.append(1)
    
    for jj in range(kk,n):
        pp.append(0)

    product=[]
    result=[]

    for k in range(0,m):
        product.append([1,1])
        for i in range(0,len(qp)):
            product0=1
            product1=1
            curQpl=qp[i]
            print(curQpl)
            for j in range(0,n):
                curidx=curQpl[j]
                test=math.pow(delta+(1.0-delta)*(1.0-math.pow(beta[k][curidx][0], q)), n*ww[k][curidx])-math.pow(1.0-math.pow(beta[k][curidx][0], q), n*ww[k][curidx])
                print(f"here {test}")
                product0*=math.pow((math.pow(delta+(1.0-delta)*(1.0-math.pow(beta[k][curidx][0], q)), n*ww[k][curidx]) +(delta*delta-1.0)*math.pow(1.0-math.pow(beta[k][curidx][0], q), n*ww[k][curidx])) /(math.pow(delta+(1.0-delta)*(1.0-math.pow(beta[k][curidx][0], q)), n*ww[k][curidx])-math.pow(1.0-math.pow(beta[k][curidx][0], q), n*ww[k][curidx])), pp[j])

                product1 *= math.pow((math.pow(delta+(1.0-delta)*math.pow(beta[k][curidx][1], q), n*ww[k][curidx]) +(delta*delta-1.0)*math.pow(beta[k][curidx][1], q*n*ww[k][curidx])) /(math.pow(delta+(1.0-delta)*math.pow(beta[k][curidx][1], q), n*ww[k][curidx]) -math.pow(beta[k][curidx][1], q*n*ww[k][curidx])), pp[j]);
            product[k][0] *= ((product0 + delta*delta - 1.0) / (product0 - 1.0));
            product[k][1] *= ((product1 + delta*delta - 1.0) / (product1 - 1.0))
            
        sumdelta1=0
        for i in range(len(pp)):
            sumdelta1+=pp[i]
        sumdelta1=1/sumdelta1
        a=[]
        a.append( math.pow((delta * math.pow(math.pow(product[k][0], factorial1(n)) - 1.0, sumdelta1)) / (math.pow(math.pow(product[k][0], factorial1(n)) + delta*delta - 1.0, sumdelta1) + (delta-1)*math.pow(math.pow(product[k][0], factorial1(n)) - 1.0, sumdelta1)), 1.0/q))                                                         
        a.append(math.pow((math.pow(math.pow(product[k][1], factorial1(n)) + delta*delta - 1.0, sumdelta1) - math.pow(math.pow(product[k][1], factorial1(n)) - 1.0, sumdelta1)) / (math.pow(math.pow(product[k][1], factorial1(n)) + delta*delta - 1.0, sumdelta1) + (delta-1)*math.pow(math.pow(product[k][1], factorial1(n)) - 1.0, sumdelta1)), 1.0/q))
        result.append(a) 
    return result

def alcoa(escm,lcm,cbr,ic):
    # column1 = [4.3817,4.0348,4.5797,4.5733,4.7176,4.4397,4.4382,4.5529,4.5990]#ESCM
    # column2 = [2.5490,2.5573,4.7917,2.5533,3.0298,3.8890,1.5092,0.6985,2.0626]#LCM
    # column3 = [0.3336,0.3712,0.2262,0.1826,0.3494,0.2594,0.2058,0.3633,0.3880]#Cost-benefit ratio
    # column4 = [35.58,33.44,53.94,51.21,29.89,70.43,70.57,0.01,0.27]#Incremental cost  
    column1=escm
    column2=lcm
    column3=cbr
    column4=ic
    ratio1=RatioOfNV(column1)
    ratio2=RatioOfNV(column2)
    ratio3=RatioOfNV(column3)
    ratio4=RatioOfNV(column4)
    print(f"ratio1 - {ratio1}")
    print(f"ratio2 - {ratio2}")
    print(f"ratio3 - {ratio3}")
    print(f"ratio4 - {ratio4}")
    MN = [
            [[ratio1[0], 1-ratio1[0]], [ratio2[0], 1-ratio2[0]], [ratio3[0], 1-ratio3[0]], [1-ratio4[0], ratio4[0]]],
            [[ratio1[1], 1-ratio1[1]], [ratio2[1], 1-ratio2[1]], [ratio3[1], 1-ratio3[1]], [1-ratio4[1], ratio4[1]]],                 
            [[ratio1[2], 1-ratio1[2]], [ratio2[2], 1-ratio2[2]], [ratio3[2], 1-ratio3[2]], [1-ratio4[2], ratio4[2]]],
            [[ratio1[3], 1-ratio1[3]], [ratio2[3], 1-ratio2[3]], [ratio3[3], 1-ratio3[3]], [1-ratio4[3], ratio4[3]]],
            [[ratio1[4], 1-ratio1[4]], [ratio2[4], 1-ratio2[4]], [ratio3[4], 1-ratio3[4]], [1-ratio4[4], ratio4[4]]],
            [[ratio1[5], 1-ratio1[5]], [ratio2[5], 1-ratio2[5]], [ratio3[5], 1-ratio3[5]], [1-ratio4[5], ratio4[5]]],
            [[ratio1[6], 1-ratio1[6]], [ratio2[6], 1-ratio2[6]], [ratio3[6], 1-ratio3[6]], [1-ratio4[6], ratio4[6]]],
            [[ratio1[7], 1-ratio1[7]], [ratio2[7], 1-ratio2[7]], [ratio3[7], 1-ratio3[7]], [1-ratio4[7], ratio4[7]]],
            [[ratio1[8], 1-ratio1[8]], [ratio2[8], 1-ratio2[8]], [ratio3[8], 1-ratio3[8]], [1-ratio4[8], ratio4[8]]],
    ]
    w = [0.250, 0.250, 0.250, 0.250]
    m=9
    n=4
    SUPP1221 = []
    SUPP1331 = []
    SUPP1441 = []    
    SUPP2332 = []
    SUPP2442 = []  
    SUPP3443 = []

    for i in range(0,m):
        SUPP1221.append(1 - distanceValue(MN[i][0][0], MN[i][1][0]))
        SUPP1331.append(1 - distanceValue(MN[i][0][0], MN[i][2][0]))
        SUPP1441.append(1 - distanceValue(MN[i][0][0], MN[i][3][0]))
        SUPP2332.append(1 - distanceValue(MN[i][1][0], MN[i][2][0]))
        SUPP2442.append(1 - distanceValue(MN[i][1][0], MN[i][3][0]))
        SUPP3443.append(1 - distanceValue(MN[i][2][0], MN[i][3][0]))

    TT1=[]
    TT2=[]
    TT3=[]
    TT4=[]
    for i in range(0,m):
        TT1.append(SUPP1221[i] + SUPP1331[i] + SUPP1441[i])
        TT2.append(SUPP1221[i] + SUPP2332[i] + SUPP2442[i])
        TT3.append(SUPP1331[i] + SUPP2332[i] + SUPP3443[i])
        TT4.append(SUPP1441[i] + SUPP2442[i] + SUPP3443[i])
    
    w1=[]
    w2=[]
    w3=[]
    w4=[]

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")

    for i in range(0,m):
        w1.append((w[0]*(1.0 + TT1[i])) / (w[0]*(1.0 + TT1[i]) + w[1]*(1.0 + TT2[i]) + w[2]*(1.0 + TT3[i]) + w[3]*(1.0 + TT4[i])))
        w2.append((w[1]*(1.0 + TT2[i])) / (w[0]*(1.0 + TT1[i]) + w[1]*(1.0 + TT2[i]) + w[2]*(1.0 + TT3[i]) + w[3]*(1.0 + TT4[i])))
        w3.append((w[2]*(1.0 + TT3[i])) / (w[0]*(1.0 + TT1[i]) + w[1]*(1.0 + TT2[i]) + w[2]*(1.0 + TT3[i]) + w[3]*(1.0 + TT4[i])))
        w4.append((w[3]*(1.0 + TT4[i])) / (w[0]*(1.0 + TT1[i]) + w[1]*(1.0 + TT2[i]) + w[2]*(1.0 + TT3[i]) + w[3]*(1.0 + TT4[i])))
        print(w1[i])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")

    ww = [
            [w1[0], w2[0], w3[0], w4[0]],
            [w1[1], w2[1], w3[1], w4[1]],
            [w1[2], w2[2], w3[2], w4[2]],
            [w1[3], w2[3], w3[3], w4[3]],
            [w1[4], w2[4], w3[4], w4[4]],
            [w1[5], w2[5], w3[5], w4[5]],
            [w1[6], w2[6], w3[6], w4[6]],
            [w1[7], w2[7], w3[7], w4[7]],
            [w1[8], w2[8], w3[8], w4[8]]
        ]

    resultOfFPWMSMHTT = FPWMSMHTT(3.0, m, n, MN, ww, 2)
    for k in range(0,m):
        index=k+1
        print(f"Beta[{index}] - {resultOfFPWMSMHTT[k][0]} - {resultOfFPWMSMHTT[k][1]}")

    domOfAggOperator=[]
    for k in range(0,m):
        domOfAggOperator.append(resultOfFPWMSMHTT[k][0])
    
    for k in range(0,m):
        index=k+1
        print(f"Beta[{index}] - {domOfAggOperator[k]}")

    tempArray=SelectionSort(domOfAggOperator)
    #Output the rank
    rank=[]
    for i in range(0,m) :
        for j in range(0,m):
            if (domOfAggOperator[j] == tempArray[i]):
                index=j+1
                rank.append(index)
                if(i==m-1):
                    print(f"A[{index}] ")
                else:
                    print(f"A[{index}] > ")
    return rank

# print(alcoa())