from sklearn.model_selection import train_test_split
import time
from preprocessing_funcs import *
from models import *
from sklearn.metrics import classification_report, confusion_matrix

num_of_repeat_same_model = 1
MaxSysCallsToProcess = 100
virus_part = {"from": 150000, "to": 152500, "rows": 2500}  # 236505 86505
numeric_c = []
output_file_name = f"convLSTM_{num_of_repeat_same_model}_times.txt"
# details
details = ["FileAttributes:", "EndOfFile:", "NumberOfLinks:", "DeletePending:", "DesiredAccess:", "Disposition:",
           "Options:", "Attributes:", "ShareMode:", "AllocationSize:", "Impersonating:", "Directory:", "IndexNumber:",
           "Access:", "Mode:", "AlignmentRequirement:", "Exclusive:", "FailImmediately:", "OpenResult:", "SyncType:",
           "PageProtection:", "Control:", "ImageBase:", "ImageSize:", "ExitStatus:", "PrivateBytes:",
           "PeakPrivateBytes:",
           "WorkingSet:", "Filter:", "PeakWorkingSet:", "ParentPID:", "Commandline:", "Currentdirectory:", "Offset:",
           "Length:",
           "Priority:", "GrantedAccess:", "Index:", "Name:", "Type:", "Data:", "Query:", "SubKeys:", "Values:",
           "HandleTags:",
           "KeySetInformationClass:", "I/OFlags:", "FileSystemAttributes:", "MaximumComponentNameLength:",
           "FileSystemName:",
           "0:00", "1:00", "2:00", "FileInformationClass:"]  # 54

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
virus_names12 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff']
virus_names15 = ['216', '3d1', '420', '42a', '46d', '52f', '56c', '5b6', '7a0', '841', '860', '9ff', 'a60', 'a8a',
                 'cbe']
virus_names1 = ["drpbx.exe"]
best_model_acc = 0
best_model_name = ''
best_model = 0

SKIP = 6
for WINDOW in [6, 9]:
    df2 = zero_padding(df, WINDOW)
    df2 = determine_target_val(df2, virus_names1).drop(['Process Name'], axis=1)  # 0 to reg, 1 to malicious
    print("dataframe length: " + str(len(df2.index)))
    X1, y1 = make_windows(df2, WINDOW, SKIP, "build")
    for NUM_IMAGES in [3, 5, 7]:
        try:
            remove_num = X1.shape[0] % NUM_IMAGES
            if remove_num != 0:
                X = X1[:remove_num * (-1)]
                y = y1[:remove_num * (-1)]
                X = X.reshape(X.shape[0] // NUM_IMAGES, NUM_IMAGES, X.shape[1], X.shape[2], X.shape[3])
                new_y = []
                count = 0
                indexes_to_del = []
                for i in range(0, len(y), NUM_IMAGES):
                    if y[i] == 0 and y[i + (NUM_IMAGES - 1)] == 0:
                        new_y.append(0)
                    elif y[i] == 1 and y[i + (NUM_IMAGES - 1)] == 1:
                        new_y.append(1)
                    else:
                        indexes_to_del.append(count)
                y = np.array(new_y)
                X = np.delete(X, indexes_to_del, axis=0)
            else:
                y = y1
                X = X1.reshape(X1.shape[0] // NUM_IMAGES, NUM_IMAGES, X.shape[1], X.shape[2], X.shape[3])

            print("--------X----------")
            print("X.shape: {}".format(X.shape))
            print("--------y----------")
            print("y.shape: {}".format(y.shape))
            print("------------------")
            # split data to train and test randomly
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # write y_test to file
            file_name = r"predictions\\" + "y_test convLSTM" + str(WINDOW) + ".txt"
            np.savetxt(file_name, y_test, delimiter='\n')

            f = open(output_file_name, "a")
            f.write(f"window:{WINDOW}, NUM_IMAGES:{NUM_IMAGES}, train len: {len(X_train)}, test len:{len(X_test)}\n******************\n")
            f.close()
            all_models = models(X.shape[1:])
            for m in all_models:
                model, title_str = m(X.shape[0])
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
                    fit_time = (fit_end_time - fit_start_time) / 60
                    print("model fit end")
                    evaluate_start_time = time.time()
                    _, accuracy = model.evaluate(X_test, y_test, verbose=2)
                    evaluate_end_time = time.time()
                    evaluate_time = (evaluate_end_time - evaluate_start_time) / 60

                    # Write result to file
                    total_res[0] += accuracy
                    total_res[1] += fit_time
                    total_res[2] += evaluate_time
                    print(title_str + f"Accuracy: {accuracy * 100}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
                    f.write(title_str + f"Accuracy: {accuracy}, fit_time={fit_time},evaluate_time={evaluate_time}\n")
                    # Create confusion_matrix
                    y_pred = (model.predict(X_test) > 0.5).astype("int32")
                    print(confusion_matrix(y_pred, y_test))
                    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
                    f.write("confusion matrix\n")
                    f.write(str((tn, fp)) + "\n")
                    f.write(str((fn, tp)) + "\n")
                    f.write("Sensitivity: " + str(int(tp)/(int(fn)+int(tp))) + "\n")
                    f.close()
                    file_name = r"predictions\\" + "prediction " + "W" + str(WINDOW) +" NUM_IMAGES:" + str(NUM_IMAGES) + " " + title_str + ".txt"
                    np.savetxt(file_name, y_pred, delimiter='\n')
                # Calc avg result
                total_res = [x / num_of_repeat_same_model for x in total_res]
                f = open(output_file_name, "a")
                f.write(f"avg: acc:{total_res[0]} train time:{total_res[1]} test time:{total_res[2]}\n")
                f.close()
                print(f"avg: acc:{total_res[0]} train time:{total_res[1]} test time:{total_res[2]}\n")
                if total_res[0] > best_model_acc:
                    best_model_acc = total_res[0]
                    best_model_name = title_str + " win size: " + str(WINDOW)+ " NUM_IMAGES:" + str(NUM_IMAGES)
                    best_model = model
        except Exception as e:
            print(e)
            print(f"filed skip={SKIP} NUM_IMAGES={NUM_IMAGES} WINDOW={WINDOW}")

print(f"best model: {best_model_name}")
# model_name = input("chose name for best_model: ")
# best_model.save("saved_models/" + best_model_name)

'''
op = int(input("for saving the best_model enter 1, else enter 0: "))
if op == 1:
    model_name = input("chose name for best_model: ")
    print("save model")
    best_model.save("saved_models/" + model_name)
'''