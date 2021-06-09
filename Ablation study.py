from sklearn.model_selection import train_test_split
import time
from preprocessing_funcs import *
from models import *
from sklearn.metrics import classification_report, confusion_matrix
num_of_repeat_same_model = 1
MaxSysCallsToProcess = 100
virus_part = {"from": 150000, "to": 153000, "rows": 3000} # 236505 86505
numeric_c = []
output_file_name = f"Ablation study.txt"
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
df1 = sort_and_cut(df1, MaxSysCallsToProcess)
# RW process
RW_input = pandas.read_csv("csv_files/v1.csv", engine='python', nrows=virus_part["to"]).tail(virus_part["rows"])
df2 = pandas.DataFrame(RW_input, columns=c)
print("benign length: " + str(len(df1.index)))
print("RW length: " + str(len(df2.index)))
# concat
df = pandas.concat([df1, df2], axis=0, join='inner').reset_index().drop(['index'], axis=1)
print("end generate data")
# start pre processing
df = separate_detail_column(df, details, "build")
df, numeric_c = norm_data(df)
df2 = W2v(df.drop(['Process Name'], axis=1), numeric_c)
df = (pandas.concat([df['Process Name'], df2], axis=1).reset_index()).drop(['index'], axis=1)
virus_names12= ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff']
virus_names15= ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff', 'a60', 'a8a', 'cbe']
virus_names1 = ["drpbx.exe"]
best_model_acc = 0
best_model_name = ''
best_model = 0

SKIP = 6
WINDOW = 15

df2 = zero_padding(df, WINDOW)
df2 = determine_target_val(df2,virus_names1).drop(['Process Name'], axis=1)  # 0 to reg, 1 to malicious
print("dataframe length: " + str(len(df2.index)))
for d in ["Operation", "Duration", "Result"]:
    df = df2.drop(d, axis=1)
    X, y = make_windows(df, WINDOW, SKIP, "build")

    print("--------X----------")
    print("X.shape: {}".format(X.shape))
    print("--------y----------")
    print("y.shape: {}".format(y.shape))
    print("------------------")
    # split data to train and test randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # write y_test to file
    file_name = r"predictions/" + f"Ablation study_y_test, droped:{d}.txt"
    np.savetxt(file_name, y_test, delimiter='\n')

    f = open(output_file_name, "a")
    f.write(f"window:{WINDOW}, train len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
    f.close()
    all_models = models(X.shape)
    for m in all_models:
        model, title_str = m()
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        total_res = [0, 0, 0]
        for _ in range(num_of_repeat_same_model):
            # Write res to file
            f = open(output_file_name, "a")
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

            # Write result to file
            total_res[0] += accuracy
            total_res[1] += fit_time
            total_res[2] += evaluate_time
            print(f"droped: {d}, Accuracy: {accuracy * 100}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
            f.write(f"droped: {d}, Accuracy: {accuracy}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
            # Create confusion_matrix
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            print(confusion_matrix(y_pred, y_test))
            tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
            f.write("confusion matrix\n")
            f.write(str((tn, fp)) + "\n")
            f.write(str((fn, tp))+ "\n")
            f.close()
            file_name = r"predictions/" + f"prediction, droped:{d}.txt"
            np.savetxt(file_name, y_pred, delimiter='\n')


