def train(model,optimizer,criterion,data,idx_train):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model,data,idx_test):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    acc = (pred[idx_test] == data.y[idx_test]).float().mean().item()
    return acc