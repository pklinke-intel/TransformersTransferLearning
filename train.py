import random
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, HerbertTokenizer


def documents_to_batch(docs, max_len):
    tokenized = tokenizer(docs)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    for i, (inp, att) in enumerate(zip(input_ids, attention_mask)):
        inp_len = len(inp)
        inp = inp[:max_len] + [PAD_TOKEN_ID] * (max_len - inp_len)
        att = att[:max_len] + [0] * (max_len - inp_len)
        input_ids[i], attention_mask[i] = inp, att
    X = torch.LongTensor(input_ids).to(DEVICE)
    ATT = torch.FloatTensor(attention_mask).to(DEVICE)
    return X, ATT


def train_on_batch(model: RobertaForSequenceClassification, optimizer, X, ATT, Y):
    model.train()
    optimizer.zero_grad()
    output = model(input_ids=X, attention_mask=ATT, labels=Y)
    loss = output["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    return loss.item()


def predict_on_batch(model: RobertaForSequenceClassification, X, ATT, Y):
    model.eval()
    output = model(input_ids=X, attention_mask=ATT, labels=Y)
    decision = output["logits"].topk(1).indices.squeeze()
    loss = output["loss"].item()
    equal = decision == Y
    correct = sum(equal).item()
    return correct, decision, loss


def prepare_data(raw_data):
    corpus = []
    labels = []
    mapping = {1: 0, 2: 0, 5: 1}
    for doc in raw_data:
        record = doc.strip().split("\t")
        if len(record) != 2:
            continue
        text, target = record
        label = int(float(target))
        if label in mapping:  # uproszczenie problemu do klasyfikacji binarnej
            corpus.append(text)
            labels.append(mapping[label])
    return corpus, labels


if __name__ == '__main__':
    with open("/data/klej/train.tsv", "r", encoding="utf8") as f:
        raw_train = f.readlines()
    with open("/data/klej/dev.tsv", "r", encoding="utf8") as f:
        raw_dev = f.readlines()

    train_corpus, train_labels = prepare_data(raw_train[1:])
    test_corpus, test_labels = prepare_data(raw_dev[1:])

    train_data = list(zip(train_corpus, train_labels))
    test_data = list(zip(test_corpus, test_labels))

    torch.manual_seed(42)
    random.seed(42)
    DEVICE = torch.device("cpu")
    tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")

    PAD_TOKEN_ID = tokenizer.pad_token_id

    # tokens = tokenizer.tokenize(train_corpus[0])
    # tokenizer(train_corpus[0])
    # tokenizer(train_corpus[0], return_tensors="pt")

    model = RobertaForSequenceClassification.from_pretrained("allegro/herbert-klej-cased-v1",
                                                             num_labels=2, hidden_dropout_prob=0.5,
                                                             attention_probs_dropout_prob=0.5)
    # outputs = model(**tokenizer(train_corpus[0], return_tensors="pt"))
    print(model)

    model = model.to(DEVICE)
    learning_rate = 0.000005
    epochs = 2
    batch_size = 10
    max_len = 120
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_train_batches = len(train_data) // batch_size + int(bool(len(train_data) % batch_size))
    num_test_batches = len(test_data) // batch_size + int(bool(len(test_data) % batch_size))

    best_acc = 0

    print('starting training loop')
    for epoch in range(epochs):    
        print('epoch: '+str(epoch))
        random.shuffle(train_data)
        total_loss = 0
        for n in tqdm(range(num_train_batches)):
            print('batch: '+str(n))
            datapoints = train_data[n * batch_size:(n + 1) * batch_size]
            documents, labels = list(zip(*datapoints))
            Y = torch.LongTensor(labels).to(DEVICE)
            X, ATT = documents_to_batch(documents, max_len)
            loss = train_on_batch(model, optimizer, X, ATT, Y)
            total_loss += loss
        print(total_loss)
        with torch.no_grad():
            total = 0
            correct = 0
            dev_loss = 0
            for n in tqdm(range(num_test_batches)):
                datapoints = test_data[n * batch_size:(n + 1) * batch_size]
                documents, labels = list(zip(*datapoints))
                Y = torch.LongTensor(labels).to(DEVICE)
                X, ATT = documents_to_batch(documents, max_len)
                result, _, loss = predict_on_batch(model, X, ATT, Y)
                dev_loss += loss
                total += batch_size
                correct += result
            acc = correct / total * 100
            print(f"acc: {acc}")
            print(f"loss: {dev_loss}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model, "herbert_ar.model")

    model = torch.load("herbert_ar.model", map_location=DEVICE)
    model.eval()

    preds = []
    for n in tqdm(range(num_test_batches)):
        datapoints = test_data[n * batch_size:(n + 1) * batch_size]
        documents, labels = list(zip(*datapoints))
        Y = torch.LongTensor(labels).to(DEVICE)
        X, ATT = documents_to_batch(documents, max_len)
        _, pred, _ = predict_on_batch(model, X, ATT, Y)
        preds.append(pred)
