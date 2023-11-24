file1 = open('MsgTraces_W5_load-50p_0.tr', 'r')
Lines = file1.readlines()

transactionDict = {}

for line in Lines:
    msgLog = line.split()

    time = float(msgLog[1])*1e-9        # in seconds
    msgSize = int(msgLog[2])#//1462*1462 # in bytes
    sender = msgLog[3]                  # ip:port
    receiver = msgLog[4]                # ip:port
    txMsgId = int(msgLog[5])

    if msgSize == 10:
        if sender in transactionDict:
            if receiver not in transactionDict[sender]:
                transactionDict[sender].append(receiver)
        else:
            transactionDict[sender] = [receiver]

for sender in transactionDict:
    print("---------------------------------------------")
    print(sender, len(transactionDict[sender]))

senderCount = len(transactionDict)
for sender in transactionDict:
    if (len(transactionDict[sender]) != senderCount - 1):
        print("Sender ", sender, " did not send requests to all the clients")
