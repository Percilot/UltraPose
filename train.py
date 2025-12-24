import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
from dataset import ConditionalDataset
from model import ConditionedUnet

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ConditionalDataset(args.dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = ConditionedUnet().to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for source, target, cond_d, cond_v in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            source, target = source.to(device), target.to(device)
            cond_d, cond_v = cond_d.to(device), cond_v.to(device)
            t = torch.randint(0, 1000, (source.size(0),), device=device).long()

            output = model(source, t, cond_d, cond_v)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for source, target, cond_d, cond_v in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                source, target = source.to(device), target.to(device)
                cond_d, cond_v = cond_d.to(device), cond_v.to(device)
                t = torch.randint(0, 1000, (source.size(0),), device=device).long()
                output = model(source, t, cond_d, cond_v)
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f}")

        if args.save_model:
            os.makedirs(args.save_model, exist_ok=True)
            save_path = os.path.join(args.save_model, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditioned UNet with dataset splitting")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_model", type=str, default="./checkpoints", help="Folder to save model checkpoints")
    args = parser.parse_args()

    train(args)
