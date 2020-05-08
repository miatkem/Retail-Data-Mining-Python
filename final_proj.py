import pandas as pd
import numpy
import itertools 
import matplotlib.pyplot as plot
from datetime import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans

#Create a histogram that describes the shopping time distribution per hour
# (please convert the date of the transaction to "%H:%M:%S" format first).
def questionA(data):
    #create array of hours
    hours=[]
    for datetime in data['InvoiceDate']:
        hms = datetime.split()[1] + ":00"
        hours.append(int(hms.split(':')[0]))

    #create histogram of hours
    plot.title('Purchasing Distribution Based on Time of Day')
    plot.hist(hours)
    plot.show()


#Find the ids of the top-10 customers who buy "coffee" related products
# (in terms of spending most money).
def questionB(data):
    #create dictionary with customerID as the key and the amount of coffee product as the value
    freqDict = {}
    for i in range(0,len(data['Description'])):
        if "COFFEE" in str(data['Description'][i]):
            if data['CustomerID'][i] != "null" and not numpy.isnan(data['CustomerID'][i]):
                cid = int(data['CustomerID'][i])
                if (cid in freqDict): 
                    freqDict[cid] += data['Quantity'][i]
                else: 
                    freqDict[cid] = data['Quantity'][i]

    #sort dictionary based on the number of coffee products
    freqDict_sorted = sorted(freqDict.items(), key=lambda kv: kv[1], reverse=True)

    #iterate through the top ten entries in the dict and print them
    it=iter(freqDict_sorted)
    print("Top-10 Customers Who Buy Coffee Related Products")
    for i in range(0,10):
        c = next(it)
        print("#" + str(i+1) + ": " + str(c[0]) + " with " + str(c[1]) + " coffee products")

#Find Top 10 best sellers for any products
def questionC(data):
    #Creat two dictionaries
    # One that uses the StockCode as the key and the description as the value
    # One that uses the StockCode as the key and the # of units sold as the value
    freqDict = {}
    descDict = {}
    for i in range(0,len(data['StockCode'])):
            if data['StockCode'][i] != "null" and data['StockCode'][i] != "POST":
                sc = str(data['StockCode'][i])
                if (sc in freqDict):
                    freqDict[sc] += data['Quantity'][i]
                else:
                    freqDict[sc] = data['Quantity'][i]
                    descDict[sc] = str(data['Description'][i])

    # sort dictionary based on the number sales per product
    freqDict_sorted = sorted(freqDict.items(), key=lambda kv: kv[1], reverse=True)

    # iterate through the top ten entries in the dict and print them
    pit=iter(freqDict_sorted)
    print("Top-10 Best Sellers")
    for i in range(0,10):
        p = next(pit)
        d = descDict[p[0]]
        print("#" + str(i+1) + ":\tStock Number: " + str(p[0]) + "\n\tDescription: " + str(d) + "\n\tSales: " + str(p[1]))

#Find the top-5 most frequent itemsets  (upto length 3) and represent their distribution
# using a bar plot/ histogram
def questionD(data):
    #create dictonary of invoiceNo pointing to an array of items in that invoice its values
    print("Compiling Transactions...")
    transactions = {}
    lastIv=''
    ivnums = data['InvoiceNo']
    for i in range(0,len(ivnums)):
        if ivnums[i] != "null" and ivnums[i][0] != 'C':
            invNum = ivnums[i]
            stockCode = data['StockCode'][i]
            if invNum == lastIv:
                transactions[invNum].append(stockCode)
            else:
                lastIv=invNum
                transactions[invNum] = []
                transactions[invNum].append(stockCode)

    #place those item arrays into another array
    # (knowing the invoice no doesnt matter at this point)
    print("Creating Itemsets...")
    itemsets = []
    transIt=iter(transactions)
    trans = next(transIt, None)
    while trans is not None:
        itemsets.append(transactions[trans])
        trans = next(transIt, None)

    #pleace data sets into a boolean matrix that accounts for sparcity
    print("Encode Transactions...")
    te = TransactionEncoder()
    oht_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(oht_ary, columns=te.columns_)

    #run aprioris algorithm on the itemssets for support >= .01 and are eith two or three items in size
    print("Executing Apriori Algorithm...")
    frequent_itemsets=apriori(sparse_df, min_support=0.01, use_colnames=True, verbose=1)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    result = frequent_itemsets[(frequent_itemsets['length'] <= 3 ) & (frequent_itemsets['length'] > 1 )]

    #sort itemssets to find the top 5 itemsets with the greatest support
    result=result.nlargest(5,'support')
    print(result)

    #plot support values of each set in bar graph
    result.plot(kind='bar', x='itemsets', y='support', rot=0)
    plot.show();

#Cluster the customers into 5 different groups based on their shopping behavior
def questionE(data):

    #Make a dictionary with CustomerID as the key and an array of tuples containing (invoiceSize, invoicePrice)
    customer = {}
    currentCustomer =data['InvoiceNo'][0]
    lastIv = data['InvoiceNo'][0]
    transactionSize=0
    transactionPrice=0
    for i in range(0, len(data['InvoiceNo'])):
        if not pd.isnull(data['CustomerID'][i]):
            if currentCustomer not in customer:
                customer[currentCustomer] = []
            if data['InvoiceNo'][i] != "null" and data['InvoiceNo'][i] != 'C':
                if data['InvoiceNo'][i] == lastIv:
                    transactionSize+=data['Quantity'][i]
                    transactionPrice += data['Quantity'][i] * data['UnitPrice'][i]
                else:
                    customer[currentCustomer].append( (transactionSize,transactionPrice) )
                    lastIv = data['InvoiceNo'][i]
                    currentCustomer = data['CustomerID'][i]
                    transactionSize = data['Quantity'][i]
                    transactionPrice = data['Quantity'][i] * data['UnitPrice'][i]
    customer[currentCustomer].append((transactionSize,transactionPrice))

    # Make X - Avg. amount of products in each invoice (per customer) AND
    # Make Y - Avg. price paid per invoice (per customer)
    xQuantity = []
    yPrice = []
    cit = iter(customer)
    cid = next(cit, None)
    while cid is not None:
        trans=customer[cid]
        translen = len(trans)
        sumQuant=0
        sumPrice=0
        for i in range(0,translen):
            sumQuant += int(trans[i][0])
            sumPrice += float(trans[i][1])
        xQuantity.append(sumQuant / translen )
        yPrice.append(sumPrice / translen )
        cid = next(cit, None)
    Data = { 'x' : xQuantity,
             'y' : yPrice }

    #Use x and y to find a k-mean cluster and print centroids
    df = pd.DataFrame(Data, columns=['x', 'y'])
    kmeans = KMeans(n_clusters=7).fit(df) #note two clusters are empty so there are really 5 clusters
    centroids = kmeans.cluster_centers_
    print(centroids)

    #create plot and color so that clusters and centroids are visible
    plot.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plot.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plot.xlabel('Customer Average Quatity of Products per Invoice')
    plot.ylabel('Customer Average Total Price per Invoice')
    plot.show()

#Find all association rules of length >=4 using a support threshold of 0.009 and confidence of 0.8
def questionF(data):
    # create dictonary of transactions pointing to an array of items
    print("Compiling Transactions...")
    transactions = {}
    lastIv = ''
    ivnums = data['InvoiceNo']
    for i in range(0, len(ivnums)):
        if ivnums[i] != "null" and ivnums[i][0] != 'C':
            invNum = ivnums[i]
            stockCode = data['StockCode'][i]
            if invNum == lastIv:
                transactions[invNum].append(stockCode)
            else:
                lastIv = invNum
                transactions[invNum] = []
                transactions[invNum].append(stockCode)

    # place those arrays of items into a and array
    print("Creating Itemsets...")
    itemsets = []

    transIt = iter(transactions)
    trans = next(transIt, None)
    while trans is not None:
        itemsets.append(transactions[trans])
        trans = next(transIt, None)

    # pleace data sets into a boolean matrix
    print("Encode Transactions...")
    te = TransactionEncoder()
    oht_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(oht_ary, columns=te.columns_)

    # run aprioris algorithm on the items sets for support >= .009
    print("Executing Apriori Algorithm...")
    frequent_itemsets = apriori(sparse_df, min_support=0.009, use_colnames=True, verbose=1)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print(frequent_itemsets)

    # run association rules algorithm to determine antecedent, consequents and confidence
    #  rules_len >= 4 AND confidence>=8
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules['consequents_len']= rules["consequents"].apply(lambda x: len(x))
    result = rules[((rules['antecedent_len'] + rules['consequents_len']) >= 4)]
    print(result)

#----------------MAIN-----------------
data = pd.read_csv("retail.csv") #open retail.csv

#recommended that one function is run at a time
#questionA(data)
questionB(data)
#questionC(data)
#questionD(data)
#questionE(data)
#questionF(data)