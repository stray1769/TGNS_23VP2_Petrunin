import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Определяем устройство для работы (GPU или CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Преобразования для изображений
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем все данные из одной папки
    full_dataset = torchvision.datasets.ImageFolder(root='./data_3',
                                                  transform=data_transforms)
    
    # Названия классов
    class_names = full_dataset.classes
    print("Найденные классы:", class_names)
    
    # Проверяем, что у нас 3 класса
    num_classes = len(class_names)
    assert num_classes == 3, "Ожидается 3 класса в данных (worms, caterpillars, slugs)"
    
    # Разделяем данные на обучающую и тестовую выборки (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Создаем загрузчики данных
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                          shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=2)
    
    # Выводим информацию о размере выборок
    print(f"Всего изображений: {len(full_dataset)}")
    print(f"Обучающая выборка: {len(train_dataset)}")
    print(f"Тестовая выборка: {len(test_dataset)}")
    
    # Функция для отображения изображений
    def imshow(inp, title=None):
        """Отображение Tensor как изображения."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
    
    # Визуализируем несколько обучающих изображений
    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    
    # Используем предобученную модель ResNet18
    model = torchvision.models.resnet18(pretrained=True)
    
    # Замораживаем все слои, кроме последнего
    for param in model.parameters():
        param.requires_grad = False
    
    # Заменяем последний слой на наш классификатор с 3 классами
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Обучение модели
    num_epochs = 10
    save_loss = []
    save_acc = []
    
    for epoch in range(num_epochs):
        print(f'Эпоха {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        model.train()  # Режим обучения
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        save_loss.append(epoch_loss)
        save_acc.append(epoch_acc.cpu().item())
        
        print(f'Потери: {epoch_loss:.4f} Точность: {epoch_acc:.4f}')
    
    # Графики обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(save_loss)
    plt.title('Функция потерь')
    plt.xlabel('Эпохи')
    
    plt.subplot(1, 2, 2)
    plt.plot(save_acc)
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.show()
    
    # Оценка модели на тестовых данных
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Точность на тестовых данных: {100 * correct / total:.2f}%')
    
    # Сохраняем модель
    torch.save(model.state_dict(), '3class_model_split.pth')
    
    # Визуализация предсказаний на тестовых данных
    def visualize_predictions(model, num_images=6):
        model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(10, 8))
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    true_label = class_names[labels[j]]
                    pred_label = class_names[preds[j]]
                    ax.set_title(f'Истинный: {true_label}\nПредсказанный: {pred_label}')
                    imshow(inputs.cpu().data[j])
                    
                    if images_so_far == num_images:
                        return
    
    visualize_predictions(model)