def run(logger, dataloader, kernel, model, argsparser):
    list_preds = []
    for k in range(3):
        logger.info("Loading dataset number {} and training the model...".format(k))
        x_train, _, y_train, _ = dataloader.get_train_val(k, 0., argsparser.rd)
        x_test = dataloader.get_test(k)
        gram_train = kernel(x_train, x_train)
        model.fit(gram_train, y_train)
        logger.info("Inference of the model...".format(k))
        gram_test = kernel(x_test, x_train)
        y_pred_test = model.predict(gram_test)
        y_pred_test[y_pred_test == -1] = 0
        list_preds += y_pred_test.tolist()

    with open("submission.csv", 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(list_preds)):
            f.write(str(i)+','+str(list_preds[i])+'\n')
