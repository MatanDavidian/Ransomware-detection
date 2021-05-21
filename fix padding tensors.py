from sklearn.model_selection import train_test_split
from preprocessing_funcs import *
from models import *
from sklearn.metrics import classification_report, confusion_matrix

MaxSysCallsToProcess = 1500
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
RW_input = pandas.read_csv("csv_files/r1.csv", engine='python')
df1 = pandas.DataFrame(RW_input, columns=c)
df1 = sort_and_cut(df1, MaxSysCallsToProcess)
# RW process
REG_input = pandas.read_csv("csv_files/v1.csv", engine='python').head(250000).tail(30000)
df2 = pandas.DataFrame(REG_input, columns=c)
# concat
df = pandas.concat([df1, df2], axis=0, join='inner').reset_index().drop(['index'], axis=1)
print("end generate data")
# start pre processing
df = separate_detail_column(df, details, "build")
df, numeric_c = norm_data(df)
df2 = W2v(df.drop(['Process Name'], axis=1), numeric_c)
df = (pandas.concat([df['Process Name'], df2], axis=1).reset_index()).drop(['index'], axis=1)
SKIP = 18

for WINDOW in [18]:
    df2 = zero_padding(df, WINDOW)
    # df2 = mean_padding(df2, MaxSysCallsToProcess) #
    df2 = determine_target_val(df2).drop(['Process Name'], axis=1)
    dataset = make_win2(df2, WINDOW, SKIP)
    DATASET_SIZE = len(dataset)
    print("DATASET_SIZE")
    print(DATASET_SIZE)
    train_size = int(0.6 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)
    print("------------------")
    print(f"window:{WINDOW}\n")
    all_models = models([0,18,57,300])
    for m in all_models[:1]:
        model, title_str = m()
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        for _ in range(1):
            # Write res to file
            model.fit(train_dataset, epochs=10, batch_size=12, verbose=2, validation_data=val_dataset)

            print("model fit end")
            _, accuracy = model.evaluate(test_dataset, verbose=2)
            print(title_str + f"Accuracy: {accuracy * 100}\n")

            import tensorflow as tf
            y_pred = model.predict(test_dataset)
            predicted_categories = tf.argmax(y_pred, axis=1)
            true_categories = tf.concat([y for x, y in test_dataset], axis=0)

            print('Confusion Matrix')
            confusion_matrix(predicted_categories, true_categories)
            classification_report(y_pred=predicted_categories, y_true=true_categories)
