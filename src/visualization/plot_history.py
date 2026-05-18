import matplotlib.pyplot as plt
import pickle

def plot_history(file_path='models_saved/train_history.pkl'):
    with open(file_path, 'rb') as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-o', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-o', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-o', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models_saved/training_report.png') # Lưu ảnh vào thư mục
    plt.show()

if __name__ == "__main__":
    plot_history()