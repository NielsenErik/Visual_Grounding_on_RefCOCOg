import torch

def save_model(model, epoch, optimizer, total_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, path+"/personal_model.pt")

def load_personal_model(model, optimizer, path):
    checkpoint = torch.load(path+"/personal_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss