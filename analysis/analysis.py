import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Constants
figsize = (10, 6)
pktPayloadSize = 1460  # Bytes
hdrSize = 40  # Bytes
torBw = 10e9  # bps
coreBw = 40e9  # bps
oneWayDel = 1.0e-6 + (pktPayloadSize + hdrSize) * 8 * (2 / torBw + 2 / coreBw)
baseRtt = oneWayDel + 1.0e-6 + 64 * 8 * (2 / torBw + 2 / coreBw)
bdpPkts = 7
saturationTime = 3.1


def removeKey(d, key):
    r = dict(d)
    del r[key]
    return r


def getPctl(a, p):
    i = int(len(a) * p)
    return a[i]


def log(filename, msg):
    with open(filename, "a") as f:
        f.write(msg + "\n")


# Final 24h runs
filenamess = []
# 144 nodes 7 vs Dynamic 50
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-50p_7_144_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-50p_dynamic_144_24h.tr",
    ]
)
# 144 nodes 7 vs Dynamic 80
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-80p_7_144_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-80p_dynamic_144_24h.tr",
    ]
)
# 144 nodes 7 vs Dynamic 100
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_7_144_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_dynamic_144_24h.tr",
    ]
)
# 144 nodes 7 vs SDN 50
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_7_144_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_sdn_144_24h.tr",
#     ]
# )
# 144 nodes 7 vs SDN 80
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-80p_7_144_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-80p_sdn_144_24h.tr",
#     ]
# )
# 144 nodes 7 vs SDN 100
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_7_144_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_base_144_24h.tr",
    ]
)
# 144 nodes 7 vs 5-7 50
# 144 nodes 7 vs 5-7 80
# 144 nodes 7 vs 5-7 100
# 144 nodes 7 vs Original 50
# 144 nodes 7 vs Original 80
# 144 nodes 7 vs Original 100
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-100p_7_8_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-100p_sdn_8_24h.tr",
#     ]
# )
# 8 nodes 7 vs Dynamic 50
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_7_8_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_dynamic_8_24h.tr",
#     ]
# )
# 8 nodes 7 vs Dynamic 80
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-80p_7_8_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-80p_dynamic_8_24h.tr",
    ]
)
# 8 nodes 7 vs Dynamic 100
filenamess.append(
    [
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_7_8_24h.tr",
        "trace/final_24hr_tests/MsgTraces_W5_load-100p_dynamic_8_24h.tr",
    ]
)
# 8 nodes 7 vs SDN 50
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_7_8_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-50p_sdn_8_24h.tr",
#     ]
# )
# 8 nodes 7 vs SDN 80
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-80p_7_8_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-80p_sdn_8_24h.tr",
#     ]
# )
# 8 nodes 7 vs SDN 100
# filenamess.append(
#     [
#         "trace/final_24hr_tests/MsgTraces_W5_load-100p_7_8_24h.tr",
#         "trace/final_24hr_tests/MsgTraces_W5_load-100p_sdn_8_24h.tr",
#     ]
# )
# 8 nodes 7 vs 5-7 50
# 8 nodes 7 vs 5-7 80
# 8 nodes 7 vs 5-7 100
# 8 nodes 7 vs Original 50
# 8 nodes 7 vs Original 80
# 8 nodes 7 vs Original 100


colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


# Verify filename matches format: trace/MsgTraces_W5_load-80p_7_8.tr
def verifyFilename(filename):
    # if filename[:18] != "trace/MsgTraces_W5":
    #     print("Filename does not match format: MsgTraces_W5")
    #     return False
    if filename[-3:] != ".tr":
        print("Filename does not match format: .tr")
        return False
    return True


def parseFilename(filename):
    # Just get the filename, not path
    return filename.split("/")[-1]


def getLoad(filename):
    return filename.split("_")[2].split("-")[1][:-1]


def getBDP(filename):
    return filename.split("_")[3]


def getNumNodes(filename):
    return filename.split("_")[4]


def analyze_filenames(filenames):
    # Initialize lists to store the results
    file_num_nodes = []
    file_network_load = []
    file_bdp = []

    for filename in filenames:
        curr_num_nodes = 0
        networkLoad = 0
        bdp = "0"

        actual_filename = parseFilename(filename)

        if verifyFilename(actual_filename):
            try:
                curr_num_nodes = int(getNumNodes(actual_filename))
                networkLoad = float(getLoad(actual_filename)) / 100
                bdp = getBDP(actual_filename)
            except Exception as e:
                print(
                    "Error parsing filename: " + actual_filename + "; Error: " + str(e)
                )

        file_num_nodes.append(curr_num_nodes)
        file_network_load.append(networkLoad)
        file_bdp.append(bdp)

    return file_num_nodes, file_network_load, file_bdp


for filenames in filenamess:
    print("\nRunning: " + str(filenames))

    # # Analyze filenames
    # file_num_nodes = []
    # file_network_load = []
    # file_bdp = []

    # for i, filename in enumerate(filenames):
    #     curr_num_nodes = 0
    #     networkLoad = 0
    #     bdp = "0"

    #     actual_filename = parseFilename(filename)

    #     if verifyFilename(actual_filename):
    #         try:
    #             curr_num_nodes = int(getNumNodes(actual_filename))
    #             networkLoad = float(getLoad(actual_filename)) / 100
    #             bdp = getBDP(actual_filename)
    #         except:
    #             print("Error parsing filename: " + actual_filename)

    #     # print(curr_num_nodes, networkLoad, bdp)
    #     file_num_nodes.append(curr_num_nodes)
    #     file_network_load.append(networkLoad)
    #     file_bdp.append(bdp)
    file_num_nodes, file_network_load, file_bdp = analyze_filenames(filenames)

    # Set constants
    NUM_NODES = file_num_nodes[0]
    if NUM_NODES == 144:
        NUM_SWITCHES = 9
        NUM_SPINES = 4
    elif NUM_NODES == 8:
        NUM_SWITCHES = 2
        NUM_SPINES = 1

    # Create output file to append to
    output_folder = f"results/{file_num_nodes[0]}nodes_{file_network_load[0]}load_{file_bdp[0]}pkts_vs_{file_bdp[1]}pkts"
    output_file = f"{output_folder}/results.txt"

    # Erase file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Create dir if doesnt exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fileMsgStartEntryDict = []
    fileMsgCompletionTimesDict = []
    fileAllMsgSizes = []
    fileMsgSizePercentiles = []

    fileMsgCompletionTimesDictSameSwitch = []
    fileMsgCompletionTimesDictDiffSwitch = []

    for i, filename in enumerate(filenames):
        # print(i, filename)

        # Dict of start time
        msgStartEntryDict = {}
        # Dict of completion time
        msgCompletionTimesDict = {}
        # List of all message sizes
        allMsgSizes = []

        msgCompletionTimesDictSameSwitch = {}
        msgCompletionTimesDictDiffSwitch = {}

        # For 144 nodes 16 switches
        def isSameSwitch(sender, receiver):
            # senderSwitchNum = int(sender.split(".")[2]) // numSwitches
            # receiverSwitchNum = int(receiver.split(".")[2]) // numSwitches
            # return senderSwitchNum == receiverSwitchNum
            # print(sender, receiver)
            senderPortNum = (int(sender.split(":")[1]) - 1000) // NUM_SWITCHES
            receiverPortNum = (int(receiver.split(":")[1]) - 1000) // NUM_SWITCHES
            # print(senderPortNum, receiverPortNum)
            return senderPortNum == receiverPortNum

        with open(filename, "r") as f:
            for line in f:
                msgLog = line.split()

                time = float(msgLog[1]) * 1e-9  # in seconds
                msgSize = int(msgLog[2])  # in bytes
                sender = msgLog[3]  # ip:port
                receiver = msgLog[4]  # ip:port
                txMsgId = int(msgLog[5])

                key = (sender, receiver, txMsgId, i)

                # Skip warmup messages
                if msgSize == 10:
                    continue

                # Sender
                if msgLog[0] == "+":
                    if key in msgStartEntryDict:
                        msgStartEntryDict[key].append(time)
                    else:
                        # Create new entry if not in dict
                        msgStartEntryDict[key] = [time]
                # Receiver
                elif msgLog[0] == "-":
                    if key in msgStartEntryDict:
                        # Find matching start entry
                        startTime = msgStartEntryDict[key].pop()

                        # Remove from startEntryDict (message completed)
                        if len(msgStartEntryDict[key]) <= 0:
                            msgStartEntryDict = removeKey(msgStartEntryDict, key)

                        # Invalid
                        if startTime < saturationTime:
                            continue

                        # Valid
                        if msgSize in msgCompletionTimesDict:
                            msgCompletionTimesDict[msgSize].append(time - startTime)
                        else:
                            # Create entry if not in dict
                            msgCompletionTimesDict[msgSize] = [time - startTime]

                        # Same switch
                        if isSameSwitch(sender, receiver):
                            if msgSize in msgCompletionTimesDictSameSwitch:
                                msgCompletionTimesDictSameSwitch[msgSize].append(
                                    time - startTime
                                )
                            else:
                                # Create entry if not in dict
                                msgCompletionTimesDictSameSwitch[msgSize] = [
                                    time - startTime
                                ]
                        # Different switch
                        else:
                            if msgSize in msgCompletionTimesDictDiffSwitch:
                                msgCompletionTimesDictDiffSwitch[msgSize].append(
                                    time - startTime
                                )
                            else:
                                # Create entry if not in dict
                                msgCompletionTimesDictDiffSwitch[msgSize] = [
                                    time - startTime
                                ]

                        # Add to list of all message sizes
                        allMsgSizes.append(msgSize)
                    else:
                        print(
                            "ERROR: Start entry of message ("
                            + sender
                            + " > "
                            + receiver
                            + ", "
                            + str(txMsgId)
                            + ") not found!"
                        )

        allMsgSizes = np.array(allMsgSizes)
        msgSizePercentiles = stats.rankdata(allMsgSizes, "max") / len(allMsgSizes) * 100

        fileMsgStartEntryDict.append(msgStartEntryDict)
        fileMsgCompletionTimesDict.append(msgCompletionTimesDict)
        fileAllMsgSizes.append(allMsgSizes)
        fileMsgSizePercentiles.append(msgSizePercentiles)

        fileMsgCompletionTimesDictSameSwitch.append(msgCompletionTimesDictSameSwitch)
        fileMsgCompletionTimesDictDiffSwitch.append(msgCompletionTimesDictDiffSwitch)

    for i, filename in enumerate(filenames):
        # print(i, filename)
        log(output_file, "\n" + str(i) + " " + filename)

        # Print same switch number of messages
        total_count = 0
        for key in fileMsgCompletionTimesDictSameSwitch[i]:
            total_count += len(fileMsgCompletionTimesDictSameSwitch[i][key])
        log(output_file, "Same switch msg count: " + str(total_count))

        # Print different switch number of messages
        total_count = 0
        for key in fileMsgCompletionTimesDictDiffSwitch[i]:
            total_count += len(fileMsgCompletionTimesDictDiffSwitch[i][key])

        log(output_file, "Different switch msg count: " + str(total_count))

    # print()

    for i, filename in enumerate(filenames):
        # print(i, filename)
        log(output_file, "\n" + str(i) + " " + filename)

        total_throughput = 0
        for msgSize in fileMsgCompletionTimesDictSameSwitch[i]:
            # if total_throughput == 0:
            #     print(msgSize, fileMsgCompletionTimesDictSameSwitch[i][msgSize])
            total_throughput += (
                msgSize
                * len(fileMsgCompletionTimesDictSameSwitch[i][msgSize])
                / sum(fileMsgCompletionTimesDictSameSwitch[i][msgSize])
            )

        # Convert from bytes/ns to bytes/s
        # total_throughput = total_throughput * 1e9

        # Convert from bytes/s to Gbps
        total_throughput = total_throughput * 10e-9 / NUM_NODES

        log(output_file, "Same switch throughput: " + str(total_throughput))

        total_throughput = 0
        for msgSize in fileMsgCompletionTimesDictDiffSwitch[i]:
            total_throughput += (
                msgSize
                * len(fileMsgCompletionTimesDictDiffSwitch[i][msgSize])
                / sum(fileMsgCompletionTimesDictDiffSwitch[i][msgSize])
            )

        # Convert from bytes/ns to bytes/s
        # total_throughput = total_throughput * 1e9

        # Convert from bytes/s to Gbps
        total_throughput = total_throughput * 10e-9 / NUM_NODES

        log(output_file, "Different switch throughput: " + str(total_throughput))

    # Basic stats

    for i, filename in enumerate(filenames):
        # print("\n", i, filename)
        log(output_file, "\n" + str(i) + " " + filename)

        # Items remaining in startEntryDict were uncompleted
        numUncompleted = 0
        for msgKey, msgStartTimes in fileMsgStartEntryDict[i].items():
            numUncompleted += len(msgStartTimes)

        log(output_file, "Number of uncompleted messages: " + str(numUncompleted))
        log(
            output_file,
            "Number of distinct message sizes: "
            + str(len(fileMsgCompletionTimesDict[i])),
        )

        cntMultMeasures = 0
        numMsgs = 0
        nonFullPacketMsgSizes = []
        for msgSize, msgCompletionTimes in fileMsgCompletionTimesDict[i].items():
            # Count number of distinct message sizes that have multiple measurements
            if len(msgCompletionTimes) > 1:
                cntMultMeasures += 1

            # Count number of distinct messages
            numMsgs += len(msgCompletionTimes)

            # Count number of non-full packet messages
            if msgSize % pktPayloadSize != 0:
                # print(msgSize%1452)
                # Dynamic packets are 1452, not 1460, so this is expected
                nonFullPacketMsgSizes.append((msgSize, msgSize % pktPayloadSize))

        log(
            output_file,
            "Number of distinct message sizes that have multiple measurements: "
            + str(cntMultMeasures),
        )
        log(output_file, "Number of distinct messages: " + str(numMsgs))

        # if nonFullPacketMsgSizes:
        #     print("Non-Full Packet Message Sizes: ", nonFullPacketMsgSizes)

    FileSimMsgSizes = []
    FileSimP50CompletionTimes = []
    FileSimP99CompletionTimes = []
    FileSimBaseCompletionTimes = []
    FileSimP50SlowDowns = []
    FileSimP99SlowDowns = []

    # Get slowdowns

    for i, filename in enumerate(filenames):
        # print("\n", i, filename)
        # log(output_file, "\n" + str(i) + " " + filename)

        SimMsgSizes = []
        SimP50CompletionTimes = []
        SimP99CompletionTimes = []
        SimBaseCompletionTimes = []
        SimP50SlowDowns = []
        SimP99SlowDowns = []

        for msgSize, msgCompletionTimes in fileMsgCompletionTimesDict[i].items():
            SimMsgSizes.append(msgSize)

            times = np.array(msgCompletionTimes)
            p50CompletionTime = np.percentile(times, 50)
            SimP50CompletionTimes.append(p50CompletionTime)
            p99CompletionTime = np.percentile(times, 99)
            SimP99CompletionTimes.append(p99CompletionTime)

            totBytes = msgSize + math.ceil(msgSize / pktPayloadSize) * (hdrSize)
            baseCompletionTime = totBytes * 8 / torBw
            if msgSize > bdpPkts * pktPayloadSize:
                baseCompletionTime += oneWayDel  # baseRtt
            else:
                baseCompletionTime += oneWayDel
            #     baseCompletionTime = (msgSize+pktPayloadSize)*8.0/10e9 + 0.5e-6 \
            #                           + 2*pktPayloadSize*8.0/40e9 + 0.5e-6

            SimBaseCompletionTimes.append(baseCompletionTime)

            SimP50SlowDowns.append(p50CompletionTime / baseCompletionTime)
            SimP99SlowDowns.append(p99CompletionTime / baseCompletionTime)

        #     slowDowns = times / baseCompletionTime
        #     slowDowns = np.where(slowDowns < 1.0, 1.0, slowDowns)

        #     SimP50SlowDowns.append(np.percentile(slowDowns,50))
        #     SimP99SlowDowns.append(np.percentile(slowDowns,99))

        zipData = sorted(
            zip(
                SimMsgSizes,
                SimP50CompletionTimes,
                SimP99CompletionTimes,
                SimBaseCompletionTimes,
                SimP50SlowDowns,
                SimP99SlowDowns,
            )
        )

        SimMsgSizes = np.array([x for x, _, _, _, _, _ in zipData])
        SimP50CompletionTimes = np.array([x for _, x, _, _, _, _ in zipData])
        SimP99CompletionTimes = np.array([x for _, _, x, _, _, _ in zipData])
        SimBaseCompletionTimes = np.array([x for _, _, _, x, _, _ in zipData])
        SimP50SlowDowns = np.array([x for _, _, _, _, x, _ in zipData])
        SimP99SlowDowns = np.array([x for _, _, _, _, _, x in zipData])

        FileSimMsgSizes.append(SimMsgSizes)
        FileSimP50CompletionTimes.append(SimP50CompletionTimes)
        FileSimP99CompletionTimes.append(SimP99CompletionTimes)
        FileSimBaseCompletionTimes.append(SimBaseCompletionTimes)
        FileSimP50SlowDowns.append(SimP50SlowDowns)
        FileSimP99SlowDowns.append(SimP99SlowDowns)

    maxSize = 0
    for i, filename in enumerate(filenames):
        # Get max size of each file
        for msgSize in fileMsgCompletionTimesDict[i].keys():
            if msgSize > maxSize:
                maxSize = msgSize

    # Create 10 xticks from 0 to maxsize
    xticks = np.arange(0, maxSize + 1, maxSize / 10)

    def moving_average(data, window_size):
        return pd.Series(data).rolling(window=window_size).mean()

    def plot_file(i, ax, window_size=50, plot_avg=False, only50=False, only99=False):
        x = [
            fileMsgSizePercentiles[i][np.where(fileAllMsgSizes[i] == msgsize)[0][0]]
            for msgsize in FileSimMsgSizes[i]
        ]
        x = FileSimMsgSizes[i]
        y50 = FileSimP50SlowDowns[i]
        y99 = FileSimP99SlowDowns[i]

        # Calculate moving averages
        moving_avg_50 = moving_average(y50, window_size=window_size)
        moving_avg_99 = moving_average(y99, window_size=window_size)

        label = f"{file_bdp[i]} pkts"

        plot_both = not only50 and not only99
        if not plot_avg:
            if only50 or plot_both:
                ax.step(x, y50, label=f"{label} 50%", color=colors[i], linestyle="--")
            if only99 or plot_both:
                ax.step(x, y99, label=f"{label} 99%", color=colors[i])

        # Plot moving averages
        if plot_avg:
            if only50 or plot_both:
                ax.plot(x, moving_avg_50, label=f"{label} 50%", color=colors[i])
            if only99 or plot_both:
                ax.plot(x, moving_avg_99, label=f"{label} 99%", color=colors[i])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("Slow Down")
    ax.set_xlabel("Message Size (Bytes)")
    ax.set_title(
        f"Homa Message Completion Slowdown for {file_num_nodes[i]} Nodes with {file_network_load[i]*100}% Load"
    )
    ax.grid(True, which="major", axis="both")

    for i, filename in enumerate(filenames):
        # print("\n", i, filename)
        plot_file(i, ax, window_size=10, plot_avg=True)  # Pass the ax object here

    ax.set_yscale("log")
    ax.set_ylim([1, 30])
    yticks = [1, 2, 3, 4, 5, 10, 20, 30]
    ax.set_yticks(yticks)
    # Put labels on y axis
    ax.set_yticklabels(yticks)
    # xlim of 5000 to inf
    ax.set_xlim([5000, maxSize])
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/slowdown.png", dpi=300)
    # plt.show()

    for i, filename in enumerate(filenames):
        # print mean of 50% and 99% slowdowns
        # print(i, filename)
        log(output_file, "\n" + str(i) + " " + filename)

        log(output_file, "50% slowdown mean: " + str(np.mean(FileSimP50SlowDowns[i])))
        log(output_file, "99% slowdown mean: " + str(np.mean(FileSimP99SlowDowns[i])))
        # log(output_file, "50% slowdown median: " + str(np.median(FileSimP50SlowDowns[i])))
        # log(output_file, "99% slowdown median: " + str(np.median(FileSimP99SlowDowns[i])))
        log(output_file, "50% slowdown std: " + str(np.std(FileSimP50SlowDowns[i])))
        log(output_file, "99% slowdown std: " + str(np.std(FileSimP99SlowDowns[i])))
        # log(output_file, "50% slowdown max: " + str(np.max(FileSimP50SlowDowns[i])))
        # log(output_file, "99% slowdown max: " + str(np.max(FileSimP99SlowDowns[i])))
        # log(output_file, "50% slowdown min: " + str(np.min(FileSimP50SlowDowns[i])))
        # log(output_file, "99% slowdown min: " + str(np.min(FileSimP99SlowDowns[i])))

    # Perform binned means, 10 bins

    res = []
    for _ in range(10):
        res.append([])

    for i, filename in enumerate(filenames):
        # Binning means
        # print(i, filename)

        stats50 = stats.binned_statistic(
            FileSimMsgSizes[i], FileSimP50SlowDowns[i], bins=10, statistic="mean"
        ).statistic
        stats99 = stats.binned_statistic(
            FileSimMsgSizes[i], FileSimP99SlowDowns[i], bins=10, statistic="mean"
        ).statistic

        for j, val in enumerate(stats50):
            res[j].append(val)
        for j, val in enumerate(stats99):
            res[j].append(val)

    # Print i, val, val for each bin
    header_str = "\n"
    for filename in filenames:
        # print(filename, end=" ")
        header_str += filename + " "
    log(output_file, header_str)

    log(output_file, "50th percentiles")
    res_str = ""
    for i, item in enumerate(res):
        # print(i, ":", item[0::2])
        res_str += str(i) + ": " + str(item[0::2]) + "\n"
    log(output_file, res_str)

    log(output_file, "99th percentiles")
    res_str = ""
    for i, item in enumerate(res):
        # print(i, ":", item[1::2])
        res_str += str(i) + ": " + str(item[1::2]) + "\n"
    log(output_file, res_str)
