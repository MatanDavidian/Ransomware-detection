from sklearn.model_selection import train_test_split
import time
from preprocessing_funcs import *
from models import *
output_file_name = "res.txt"

MaxSysCallsToRegProcess = 10
virus_part = {"from": 150000, "to": 150050, "rows": 50}  # 236505 86505
numeric_c = []
# details
details = ["FileAttributes:","EndOfFile:","NumberOfLinks:","DeletePending:","DesiredAccess:","Disposition:",
     "Options:","Attributes:","ShareMode:","AllocationSize:","Impersonating:","Directory:","IndexNumber:",
     "Access:","Mode:","AlignmentRequirement:","Exclusive:","FailImmediately:","OpenResult:","SyncType:",
     "PageProtection:","Control:","ImageBase:","ImageSize:","ExitStatus:","PrivateBytes:","PeakPrivateBytes:",
     "WorkingSet:","Filter:","PeakWorkingSet:","ParentPID:","Commandline:","Currentdirectory:","Offset:","Length:",
     "Priority:","GrantedAccess:","Index:","Name:","Type:","Data:","Query:","SubKeys:","Values:","HandleTags:",
     "KeySetInformationClass:","I/OFlags:","FileSystemAttributes:","MaximumComponentNameLength:","FileSystemName:",
     "0:00","1:00","2:00","FileInformationClass:"]  # 54

c = ["Process Name", "Operation", "Duration", "Result", "Detail"] + details + ["malicious"]
# benign processes
REG_input = pandas.read_csv("csv_files/r1.csv", engine='python')
df1 = pandas.DataFrame(REG_input, columns=c)
df1 = sort_and_cut(df1, MaxSysCallsToRegProcess)

# RW process
if virus_part["rows"] != 0:
    RW_input = pandas.read_csv("csv_files/v1.csv", engine='python', nrows=virus_part["to"]).tail(virus_part["rows"])
else:
    RW_input = pandas.read_csv("csv_files/v1.csv", engine='python')

df2 = pandas.DataFrame(RW_input, columns=c)

# print dfs len
print("benign length: " + str(len(df1.index)))
print("RW length: " + str(len(df2.index)))
# concat
df = pandas.concat([df1, df2], axis=0, join='inner').reset_index().drop(['index'], axis=1)
print("end generate data")
# start pre processing
df = separate_detail_column(df, details, "build")
df, numeric_c = norm_data(df)
df2 = W2v(df.drop(['Process Name'], axis=1), numeric_c)
print("end W2V")


virus_names1 = ['drpbx.exe']
virus_names12 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff']
virus_names15 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff', 'a60', 'a8a', 'cbe']

df = (pandas.concat([df['Process Name'], df2], axis=1).reset_index()).drop(['index'], axis=1)
df = determine_target_val(df, virus_names1).drop(['Process Name'], axis=1)

# 1D Models -----------------------
y = df["malicious"].values.tolist()
y = np.array(y).astype(np.float32)
X = df.drop('malicious', axis=1)
X = X.values.tolist()
X = np.asarray(X, dtype="float32")
#X = X.reshape(X.shape[0], -1)

print("--------X----------")
print("X.shape: {}".format(X.shape))
print("--------y----------")
print("y.shape: {}".format(y.shape))
print("------------------")
# split data to train and test randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_t1 = y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_t2 = y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_t3 = y_test
print("--- y1 ---")
print(y_t1)
print("--- y2 ---")
print(y_t2)
print("--- y3 ---")
print(y_t3)
X = 0
esdf()
####################################################################
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(57, 300)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
fit_start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=12, verbose=2)
fit_end_time = time.time()
fit_time = (fit_end_time - fit_start_time)/60
print("model fit end")
evaluate_start_time = time.time()
_, accuracy = model.evaluate(X_test, y_test, verbose=2)
evaluate_end_time = time.time()
evaluate_time = (evaluate_end_time - evaluate_start_time)/60

f = open(output_file_name, "a")
f.write(f"CNN2: 128,flat,100,1\ntrain len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
f.write(f"acc:{accuracy}, train time:{fit_time}, test time:{evaluate_time}\n")
f.close()


####################################################################

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(57, 300)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
fit_start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=12, verbose=2)
fit_end_time = time.time()
fit_time = (fit_end_time - fit_start_time)/60
print("model fit end")
evaluate_start_time = time.time()
_, accuracy = model.evaluate(X_test, y_test, verbose=2)
evaluate_end_time = time.time()
evaluate_time = (evaluate_end_time - evaluate_start_time)/60

f = open(output_file_name, "a")
f.write(f"DNN: 128,34,10,flat,10,1\ntrain len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
f.write(f"acc:{accuracy}, train time:{fit_time}, test time:{evaluate_time}\n")
f.close()

# ---------------------------------

####################################################################
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(57, 300)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
fit_start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=12, verbose=2)
fit_end_time = time.time()
fit_time = (fit_end_time - fit_start_time)/60
print("model fit end")
evaluate_start_time = time.time()
_, accuracy = model.evaluate(X_test, y_test, verbose=2)
evaluate_end_time = time.time()
evaluate_time = (evaluate_end_time - evaluate_start_time)/60

f = open(output_file_name, "a")
f.write(f"CNN1: 64,128,flat,1000,500,100,1\ntrain len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
f.write(f"acc:{accuracy}, train time:{fit_time}, test time:{evaluate_time}\n")
f.close()
