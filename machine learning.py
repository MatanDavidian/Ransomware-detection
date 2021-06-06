from sklearn.model_selection import train_test_split
import time
from preprocessing_funcs import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
MaxSysCallsToRegProcess = 5000
virus_part = {"from": 150000, "to": 236505, "rows": 86505}  # 236505 86505
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
df = (pandas.concat([df['Process Name'], df2], axis=1).reset_index()).drop(['index'], axis=1)
SKIP = 18

best_model_acc = 0
best_model_name = ''
best_model = 0

virus_names1 = ['drpbx.exe']
virus_names12 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff']
virus_names15 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff', 'a60', 'a8a', 'cbe']


df = determine_target_val(df, virus_names1).drop(['Process Name'], axis=1)  # 0 to reg, 1 to malicious

y = df["malicious"].values.tolist()
y = np.array(y).astype(np.float32)
X = df.drop('malicious', axis=1)
X = X.values.tolist()
X = np.asarray(X, dtype="float32")
X = X.reshape(X.shape[0], -1)

print("--------X----------")
print("X.shape: {}".format(X.shape))
print("--------y----------")
print("y.shape: {}".format(y.shape))
print("------------------")
# split data to train and test randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X = 0
df = 0

print("train_test_split end")



# SVM linear
print("start svm")
fit_start_time = time.time()
S_model = LinearSVC(random_state=1, max_iter=1000).fit(X_train, y_train)  # kernel='poly'
fit_end_time = time.time()
print("SVC fit end")
evaluate_start_time = time.time()
S_res = S_model.score(X_test, y_test)
evaluate_end_time = time.time()
print(f"The SVM linear func result: {S_res} fit time:{fit_end_time-fit_start_time}, evaluate time:{evaluate_end_time-evaluate_start_time}")

# SVM poly
fit_start_time = time.time()
S_model = SVC(kernel='poly',random_state=1, max_iter=1000).fit(X_train, y_train)
fit_end_time = time.time()
print("SVC fit end")
evaluate_start_time = time.time()
S_res = S_model.score(X_test, y_test)
evaluate_end_time = time.time()
print(f"The SVM poly func result: {S_res} fit time:{fit_end_time-fit_start_time}, evaluate time:{evaluate_end_time-evaluate_start_time}")



fit_start_time = time.time()
F_model = RandomForestClassifier().fit(X_train, y_train)
fit_end_time = time.time()
print("F_model fit end")
evaluate_start_time = time.time()
F_res = F_model.score(X_test, y_test)
evaluate_end_time = time.time()
print(f"The Random Forest result: {F_res} fit time:{fit_end_time-fit_start_time}, evaluate time:{evaluate_end_time-evaluate_start_time}")