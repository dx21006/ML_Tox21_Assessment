from sklearn import metrics
def get_model_metrics(fitted_model, x_train, x_val, y_train, y_val):
    pred_X_train = fitted_model.predict(x_train)
    pred_X_val = fitted_model.predict(x_val)
    train_bal_acc = metrics.balanced_accuracy_score(y_train, pred_X_train)
    val_bal_acc = metrics.balanced_accuracy_score(y_val, pred_X_val)
    train_recall = metrics.recall_score(y_train, pred_X_train)
    val_recall = metrics.recall_score(y_val, pred_X_val)
    train_precision = metrics.precision_score(y_train, pred_X_train)
    val_precision = metrics.precision_score(y_val, pred_X_val)
    train_f1 = metrics.f1_score(y_train, pred_X_train)
    val_f1 = metrics.f1_score(y_val, pred_X_val)
    metric_list = [train_bal_acc, train_recall, train_precision, train_f1, val_bal_acc, val_recall, val_precision, val_f1]
    print(metric_list)