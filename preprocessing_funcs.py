import pandas
import re
import fasttext.util
import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import tensorflow as tf

ft = fasttext.load_model('cc.en.300.bin')
print("end load fast text model")


def determine_target_val(df):
    for index, l in df.iterrows():
        if l["Process Name"] == "drpbx.exe":
            df.at[index, "malicious"] = '1'
        elif l["Process Name"] == "":
            print("no Process Name")
            exit()
        else:
            df.at[index, "malicious"] = '0'
    return df


def separate_detail_column(df, details, option):
    for col_name in df.columns:
        df[col_name] = df[col_name].astype(str)
    i = 0
    for index, l in df.iterrows():
        i += 1
        if i % 10000 == 0:
            print(index)
        op = (l["Operation"]).replace("Reg", "Registry")
        op = re.sub('([A-Z])', r' \1', op)[1:]
        df.at[index, "Operation"] = op
        if pandas.notnull(l["Detail"]):
            rowDetail = l["Detail"]
            first_arg = rowDetail.find(':')
            detailsList = re.findall(',[^,]*?:', rowDetail)
            # TODO DeviceIoControl has only 1 arg, with :' '
            # all start indexes
            start_end_index = list(map(lambda x: [x[2:], rowDetail.find(x[2:])], detailsList))
            # add first attribute
            start_end_index.insert(0, [rowDetail[:first_arg + 1], 0])
            # add last indexes
            for x in start_end_index:
                x.append(x[1] + len(x[0]))
            # print(start_end_index)
            # insert args in separate columns
            for x in range(len(start_end_index) - 1):
                if start_end_index[x][0] in details:
                    value = (rowDetail[start_end_index[x][2]+1:start_end_index[x + 1][1] - 2]).replace(',', '')
                    ret_val = to_number(value)
                    value = ret_val[1]
                    # separate connected words
                    if not ret_val[0] and len(value) > 4:
                        space_sum = sum(1 for c in value if c == ' ')
                        if space_sum == 0 and value[0].isupper() and value[1].islower():
                            # print(value)
                            upcase_sum = sum(1 for c in value if c.isupper())
                            lowercase_sum = sum(1 for c in value if c.islower())
                            if upcase_sum > 1 and lowercase_sum > 2:
                                # print(value)
                                value = re.sub('([A-Z])', r' \1', value)[1:]
                                # print(value)
                    df.at[index, start_end_index[x][0]] = value
            if len(start_end_index) != 0 and start_end_index[-1][0] in details:
                value = (rowDetail[start_end_index[-1][2]+1:]).replace(',', '')
                value = to_number(value)[1]
                #print(value)
                df.at[index, start_end_index[-1][0]] = value
    print("end separate detail column")

    # df.iloc[:, -1] = '0'
    del df["Detail"]
    return df


def to_number(s):
    try:
        f = float(s)
        return True, f
    except ValueError:
        try:
            i = int(s, 0)
            return True, i
        except ValueError:
            return False, s


def W2v(df, numeric_c):
    for col_name in df.columns:
        df[col_name] = df[col_name].astype(str)
    empty_list = [0] * 299
    t_empty_list = [0] * 300
    for index, l in df.iterrows():
        for col_name in df.columns:
            if col_name == 'malicious':
                continue
            value = l[col_name]
            if value in ["nan", "0.0"]:
                df.at[index, col_name] = t_empty_list
            elif col_name not in numeric_c:
                #print("-------------")
                #print(col_name)
                df.at[index, col_name] = ret_vec(value)
            else:
                df.at[index, col_name] = [df.at[index, col_name]] + empty_list
        if index % 10000 == 0:
            print(f"index: {index}")
    return df


def mean_padding(df, Pad):
    # Change zero padding to mean padding
    print(df)
    count = 0
    for index, l in df.iterrows():
        if l["Operation"][0] == 0:
            #print(f"index: {index}")
            #print(f"count:{count}")
            temp_df = df[index - count: index]
            #print(temp_df)
            #print("#######")
            count = 0
            mean_array = []
            for col_name in temp_df.columns:
                #print(col_name)
                #print(temp_df[col_name])
                if col_name != "malicious":
                    temp = temp_df[col_name].values.tolist()
                    temp = np.asarray(temp, dtype=np.float32)
                    temp = np.mean(temp, axis=0).tolist()
                    mean_array.append(temp)
            for index2, l2 in df[index: index + Pad].iterrows():
                if l2['Operation'][0] == 0:
                    count -= 1
                    col_num = 0
                    df.at[index2, "malicious"] = 0
                    for col_name in temp_df.columns:
                        if col_name != "malicious":
                            df.at[index2, col_name] = mean_array[col_num]
                            col_num += 1
                else:
                    break
        if count == Pad:
            count = 0
        count += 1
    print("end padding")
    return df


def zero_padding(df, win_size):
    print("start zero_padding")
    df = df.set_index(['Process Name', df.groupby('Process Name').cumcount()])
    indexes = []
    for name, group in df.groupby('Process Name'):
        pad = win_size - (len(group) % win_size)
        indexes.append(len(group) + pad)

    names = []
    for i in range(len(indexes)):
        name = [df.index.levels[0][i]] * indexes[i]
        names += name
    indexes = [range(i) for i in indexes]
    indexes = [list(i) for i in indexes]
    # flatten
    indexes = [item for sublist in indexes for item in sublist]

    arr = [names, indexes]
    mux = pandas.MultiIndex.from_arrays(arr, names=df.index.names)
    df = df.reindex(mux, fill_value=[0] * 300).reset_index(level=1, drop=True).reset_index()
    print("end zero_padding")
    return df


def sort_and_cut(df, max_sc):
    df = df.groupby('Process Name').head(max_sc)
    return df.sort_values(['Process Name'], kind='mergesort')


def norm_data(df):
    numeric_c = []
    for c in df.columns:
        df[c] = df[c].replace(['nan', 'n/a'], 0, regex=True)
    for c in df.columns:
        if c == 'malicious':
            continue
        try:
            df[c] = pandas.to_numeric(df[c])
            # print(c)
            min_c = df[c].min()
            df[c] = np.float32((df[c] - min_c) / (df[c].max() - min_c))
            numeric_c.append(c)
        except:
            pass
    print("end normalized")
    return df, numeric_c


def ret_vec(value):
    vec = re.split(',| |_|-', value)

    vec = [x for x in vec if x]
    if vec:
        #print(vec)
        vec300 = list(map(lambda x: ft.get_word_vector(x), vec))
        #print(vec300)
        meanVec = np.mean(vec300, axis=0)
        return meanVec
    return [0] * 300


# https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
def make_windows(df, WINDOW, SKIP, option):
    print("start make windows")

    def indexing(df, y_val):
        r_num = len(df.index)
        if option == "build":
            X = df.drop('malicious', axis=1).values.tolist()
        else:
            X = df.values.tolist()
        X = np.asarray(X, dtype="float32")

        num_of_windows = r_num // SKIP - WINDOW // SKIP - 1
        indexer = np.arange(WINDOW)[None, :] + SKIP * np.arange(num_of_windows)[:, None]

        X = X[indexer]
        if option == "build":
            y = [y_val] * num_of_windows
            y = np.array(y)
            return X, y
        else:
            return X
    if option == "build":
        X1, y1 = indexing(df[df["malicious"] == '0'], 0)
        print(f"X1.shape[0] = {X1.shape[0]}")
        X2, y2 = indexing(df[df["malicious"] == '1'], 1)
        print(f"X2.shape[0] = {X2.shape[0]}")
        print("end make windows")
        try:
            return np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0)
        except:
            if X1.shape[0] > X2.shape[0]:
                return X1, y1
            else:
                return X2, y2
    else:
        X = indexing(df, 0)
        print(f"X.shape[0] = {X.shape[0]}")
        print("end make windows")
        return X


def make_win2(df, WINDOW, SKIP):
    print("start make win2")
    y = df['malicious'].values.tolist()
    y = np.asarray(y, dtype="float32")
    y = tf.one_hot(y, 2)
    print("y:", y)
    X = df.drop('malicious', axis=1).values.tolist()
    X = np.asarray(X, dtype="float32")
    print("end to numpy")
    dataset = timeseries_dataset_from_array(data=X, targets=y, sequence_length=WINDOW,
                                                                   sampling_rate=1, sequence_stride=SKIP,
                                                                   batch_size=12, shuffle=True)
    print("end make win2")
    return dataset


