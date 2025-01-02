import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score


class EmotionDataset(Dataset):
    def __init__(self, image_folder, label_mapping, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for label_name in os.listdir(image_folder):
            label_path = os.path.join(image_folder, label_name)
            for img_name in os.listdir(label_path):
                self.image_paths.append(os.path.join(label_path, img_name))
                self.labels.append(label_mapping[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_metrics(preds, labels):
    preds = preds.argmax(dim=1).cpu()
    labels = labels.cpu()
    accuracy = (preds == labels).float().mean().item()

    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    f1 = f1_score(labels, preds, average='weighted')

    return accuracy, precision, f1


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device="cpu"):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            acc, precision, f1 = compute_metrics(outputs.logits, labels)
            train_accuracy += acc

            progress_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0
        val_accuracy = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in val_progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                acc, precision, f1 = compute_metrics(outputs.logits, labels)
                val_accuracy += acc
                all_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(precision)
        val_f1_scores.append(f1)

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}, Val Precision = {precision:.4f}, "
            f"Val F1 Score = {f1:.4f}")

        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

    final_model_path = "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_f1_scores)
    print("Training complete!")

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    train_path = r"C:\Users\Abhijith lappy\PycharmProjects\Emotion Detector Live\Dataset\train"
    test_path = r"C:\Users\Abhijith lappy\PycharmProjects\Emotion Detector Live\Dataset\test"

    batch_size = 64
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = EmotionDataset(train_path, label_mapping, transform=transform)
    val_dataset = EmotionDataset(test_path, label_mapping, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=7)
    model.to(device)

    optimizer = optim.Adam([
        {'params': model.vit.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, device=device)

    model_save_path = r"C:\Users\Abhijith lappy\PycharmProjects\Emotion Detector Live\vit_emotion_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
