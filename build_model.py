from sklearn.model_selection import train_test_split
import time
from preprocessing_funcs import *
from models import *
input_csv_file = "csv_files/100k_pad1.5k.CSV"
Pad = 1500
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
csv_input = pandas.read_csv("csv_files/end_100k_no_SD.csv", engine='python')
df = pandas.DataFrame(csv_input, columns=c)
df = separate_detail_column(df, details, "build")
del df["Process Name"]
df, numeric_c = norm_data(df)
df = W2v(df, numeric_c)
df = mean_padding(df, Pad)

SKIP = 20
f = open(f"all_models_5_times.txt", "a")
f.write(f"SKIP:{SKIP}\n*********\n")
f.close()
best_model_acc = 0
best_model_name = ''
best_model = 0
for WINDOW in [6]:
    X, y = make_windows(df, WINDOW, SKIP, "build")
    print("--------X----------")
    print("X.shape: {}".format(X.shape))
    print("--------y----------")
    print("y.shape: {}".format(y.shape))
    print("------------------")
    # split data to train and test randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    f = open(f"all_models_5_times.txt", "a")
    f.write(f"window:{WINDOW}, train len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
    f.close()
    all_models = models(X.shape)
    for m in all_models:
        model, title_str = m()
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        total_res = [0, 0, 0]
        for _ in range(5):
            # Write res to file
            f = open(f"all_models_5_times.txt", "a")
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
            total_res[0] += accuracy
            total_res[1] += fit_time
            total_res[2] += evaluate_time
            print(title_str + f"Accuracy: {accuracy * 100}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
            f.write(title_str + f"Accuracy: {accuracy}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
            f.close()
        total_res = [x / 5 for x in total_res]
        f = open(f"all_models_5_times.txt", "a")
        f.write(f"avg: acc:{total_res[0]} train time:{total_res[1]} test time:{total_res[2]}\n")
        f.close()
        print(f"avg: acc:{total_res[0]} train time:{total_res[1]} test time:{total_res[2]}\n")
        if total_res[0] > best_model_acc:
            best_model_acc = total_res[0]
            best_model_name = title_str + " win size: " + str(WINDOW)
            best_model = model

print(f"best model: {best_model_name}")
op = int(input("for saving the best_model enter 1, else enter 0: "))
if op == 1:
    model_name = input("chose name for best_model: ")
    print("save model")
    best_model.save("saved_models/" + model_name)