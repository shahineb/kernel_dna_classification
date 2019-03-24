

def run(logger, dataloader, kernel, model, argsparser):
    for k in range(3):
        logger.info("Loading dataset number {} and training the model...".format(k))
        x_train, x_val, y_train, y_val = dataloader.get_train_val(k, argsparser.val_size, argsparser.rd)
        gram_train = kernel(x_train, x_train)
        model.fit(gram_train, y_train)
        logger.info("Evaluation of the model...".format(k))
        y_pred_train = model.predict(gram_train)
        model.evaluate(y_train, y_pred_train)
        gram_val = kernel(x_val, x_train)
        y_pred_val = model.predict(gram_val)
        model.evaluate(y_val, y_pred_val, val=True)