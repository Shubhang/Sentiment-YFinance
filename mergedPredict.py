import tflearn
import string
import pickle
import argparse
import numpy as nm
import os
import pandas as pd

from selenium import webdriver
import time


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from webdriver_manager.chrome import ChromeDriverManager


options = webdriver.ChromeOptions()
options.add_argument("--headless")
# options.add_argument("--start-maximized")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
#driver = webdriver.Chrome(options=options)

options.accept_untrusted_certs = True
options.assume_untrusted_cert_issuer = True
# chrome configuration
# More: https://github.com/SeleniumHQ/docker-selenium/issues/89
# And: https://github.com/SeleniumHQ/docker-selenium/issues/87
options.add_argument("incognito")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1024,800")
options.add_argument("disable-extensions")
options.add_argument("--start-maximized")
options.add_argument("--test-type=browser")
options.add_argument("--disable-impl-side-painting")
options.add_argument("--disable-setuid-sandbox")
options.add_argument("--disable-seccomp-filter-sandbox")
options.add_argument("--disable-breakpad")
options.add_argument("--disable-client-side-phishing-detection")
options.add_argument("--disable-cast")
options.add_argument("--disable-cast-streaming-hw-encoding")
options.add_argument("--disable-cloud-import")
options.add_argument("--disable-popup-blocking")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-session-crashed-bubble")
options.add_argument("--disable-ipv6")
options.add_argument("--allow-http-screen-capture")
driver = webdriver.Chrome(ChromeDriverManager().install(), options = options)


def convertTimeStamps(timeStamps):
    newTimeStamps = []

    for j in range(0, len(timeStamps)):
        hours, minutes, num, days = 0,0,0,0
        x = timeStamps[j]
        elements = x.split()
        if(elements[-1] == "yesterday"):
            hours = 24
            minutes = hours * 60
        elif(elements[-1] == "ago"):
            num = int(elements[0])
            if(elements[1] == 'hour' or elements[1] == 'hours'):
                hours = num
                minutes = hours * 60
            elif(elements[1] == 'minute' or elements[1] == 'minutes'):
                minutes = num
            elif(elements[1] == 'day' or elements[1] == 'days'):
                hours = 24 * num
                minutes = hours * 60
            elif(elements[1] == 'second' or elements[1] == 'seconds'):
                minutes = 0
            elif(elements[1] == 'month' or elements[1] == 'months'):
                hours = 24 * 30 * 1
                minutes = hours * 60
            elif(elements[1] == 'year' or elements[1] == 'years'):
                days = 365 * num
                hours = days * 24
                minutes = hours * 60
        elif(elements[-2] == "last"):
            hours = 24 * 30 * 1
            minutes = hours * 60
        #newTimeStamps.append("{}:{}".format(minutes, x))
        newTimeStamps.append(minutes)
    return newTimeStamps
    #print(newTimeStamps)
    #print(timeStamps)

def getConvo(ticker):

    '''
    :param ticker: The ticker symbol extracted from greg.py
    :return: Lists containing both the comments and relative timestamps from Yahoo Finance Conversations
    '''

    commentList = []
    stamps = []
    driver.get('https://finance.yahoo.com/quote/'+str(ticker)+'/community?p='+str(ticker))
    showMore = driver.find_element_by_xpath('//*[@id="canvass-0-CanvassApplet"]/div/button')
    for _ in range(5):
        showMore.click()
        time.sleep(2)
    comments = driver.find_elements_by_class_name('comment')
    for i in range(len(comments)):
        stamps.append(driver.find_element_by_xpath('//*[@id="canvass-0-CanvassApplet"]/div/ul/li['+str(i+1)+']/div/div[1]/span/span').text)
    for c in comments:
        comment = c.text
        comment = comment[:comment.rfind('Reply')].replace('\n',' ')
        if 'ago' in comment:
            ago = comment.find('ago')
            commentList.append(comment[ago+4:])
        if 'Yesterday' in comment:
            yesterday = comment.find('yesterday')
            commentList.append(comment[yesterday+10:])

    stampsOut = []
    for i in range(len(commentList)):
        #print(commentList[i])
        stampsOut.append(stamps[i])

    realStampsOut = convertTimeStamps(stampsOut)
    # print("Stamps out are as follows: {}".format(stampsOut))
    # print(commentList)
    result = []

    for i in range(0,len(commentList)):
        result.append([commentList[i],realStampsOut[i], stampsOut[i]])

    z = sorted(result, key = lambda x: x[1])

    #print(result)
    comments1, stamps1, rightstamp1 = [], [], []
    for kl in range(0, len(z)):
        comments1.append(z[kl][0])
        stamps1.append(z[kl][2])
        rightstamp1.append(z[kl][1])
    return comments1, stamps1


try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

def convertTextToIndex(dictionary, text):
    document = []
    text = text.lower().encode('utf-8')
    words = text.split()
    for word in words:
        word = word.translate(None, string.punctuation.encode('utf-8'))
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        document.append(index)

    ln = 150 - len(document)
    if ln>0 :
        document = nm.pad(document, (0, ln), 'constant')
    return document

def DoStuff(companyTICKER):

        # parser = argparse.ArgumentParser(description='Train with lstm')
        # parser.add_argument('language')
        # parser.add_argument('text')
        #
        # args = parser.parse_args()
        # lang = args.language
        # text = args.text
        lang = "en"
        res,resPosenti = [], []
        #company = input("Company ticker: ")
        company = companyTICKER
        convos, times = getConvo(company)
        f = open('./dictionaries/'+lang+'dictionary.pickle', 'rb')
        dictionary = pickle.load(f)
        f.close()
        net = tflearn.input_data([None, 150])
        net = tflearn.embedding(net, input_dim=10000, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                     loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.load("checkpoints/"+lang+"/"+lang+"tf.tfl")
        savedStuffList = []
        sum,count = 0, 0
        for i in range(len(convos)):
            text = convos[i][:149]

            result = model.predict([convertTextToIndex(dictionary, text)])
            temp = [result[0][0], result[0][1]]
            res.append(temp)

            posenti = result[0][0]
            resPosenti.append(posenti)
            sum += posenti
            count += 1
            savedStuffList.append([text, posenti])
            # print("negative="+str(result[0][0]))
            # print("positive="+ str(result[0][1]))

        fileName = company + ".txt"
        with open(fileName, 'w') as f:
            for item in savedStuffList:
                f.write("%s\n" % item)
        #print(resPosenti)
        #print(type(resPosenti))
        mean = sum/count
        print("General Sentiment:",mean)
        times.reverse()
        resPosenti.reverse()
        #llst = [times, resPosenti]
        #sentiPlot = pd.DataFrame(times, resPosenti, columns =['Time', 'Sentiment'])
        #sentiPlot = pd.DataFrame(list(zip(times, times, resPosenti)), columns =['Time', 'Time1', 'Sentiment', ])
        sentiPlot = pd.DataFrame(list(zip(times, resPosenti)), columns =['Time', 'Sentiment'])
        # print(list(sentiPlot.columns))
        # print(sentiPlot.head())
        # sentiPlot.set_index('Time', inplace=True)
        sentiPlot = sentiPlot.groupby('Time', sort=False).mean()
        # print(" ")
        # print(list(sentiPlot.columns))
        print(sentiPlot.head())
        #sentiPlot.columns = ['Time', 'Sentiment']
        # sentiPlot.plot(kind = 'line', x = 'Time', y = 'Sentiment')
        csvName = str(company) + ".csv"
        sentiPlot.to_csv(csvName, sep='\t')

        #
        # plt.xlabel('Time')
        # plt.xticks(rotation=60)
        # plt.ylabel('Sentiment')
        # #plt.yscale('log')
        # plt.grid(True)
        # plt.title('Sentiment vs Time for: {}'.format(company))
        # print("Index is {}".format(sentiPlot.index))
        # plt.plot(sentiPlot.index, sentiPlot['Sentiment'])
        # #plt.plot(sentiPlot.index, sentiPlot['Time'], sentiPlot['Sentiment'])
        # #plt.scatter(sentiPlot)
        # imgname = str(company) + '.jpeg'
        # plt.savefig(imgname, dpi=2000, bbox_inches='tight')
        # plt.show()
        # print(times)
        # print("Total analyzed!: {}".format(count))


if __name__ == '__main__':
    #x = ['MSFT']

    DoStuff('MSFT')
    DoStuff('TSLA')
    DoStuff('AAPL')
